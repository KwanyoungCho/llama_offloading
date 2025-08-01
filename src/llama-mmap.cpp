#include "llama-mmap.h"

#include "llama-impl.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>

/*
 * llama-mmap.cpp: 모델 파일 메모리 매핑 시스템
 * 
 * 이 파일은 세 가지 핵심 클래스를 제공합니다:
 * 
 * 1. llama_file - 크로스플랫폼 파일 I/O (Windows/Linux/Unix)
 * 2. llama_mmap - 모델 파일의 효율적인 메모리 매핑
 * 3. llama_mlock - 메모리 잠금으로 스와핑 방지
 * 
 * 주요 사용 목적:
 * - 대용량 모델 파일(수 GB~수십 GB)의 효율적인 로딩
 * - 메모리 매핑을 통한 가상 메모리 활용
 * - 모델 가중치 접근 시 운영체제 캐싱 최적화
 * - 필요 시 부분적 메모리 해제 지원
 * 
 * 언제 사용되는가:
 * - llama_model_loader에서 모델 파일(.gguf) 로드 시
 * - 큰 텐서 데이터를 직접 메모리에 복사하지 않고 포인터로 접근
 * - SSD/HDD에서 RAM으로의 데이터 이동 최적화
 */

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// TODO: consider moving to llama-impl.h if needed in more places
#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

// ============================================================================
// llama_file: 크로스플랫폼 파일 I/O 추상화
// ============================================================================
/*
 * 목적: Windows와 Linux/Unix에서 통일된 파일 I/O 인터페이스 제공
 * 특징:
 * - 동기적 I/O (블로킹)
 * - 청크 단위 읽기/쓰기 (64MB 단위로 분할)
 * - 크로스플랫폼 에러 처리
 * - 대용량 파일 지원 (64bit 오프셋)
 */

struct llama_file::impl {
#if defined(_WIN32)
    // Windows용 구현: Win32 API 사용
    HANDLE fp_win32;
    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

    impl(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    // 중요: 64MB 청크 단위로 읽기 (대용량 파일 최적화)
    void read_raw(void * ptr, size_t len) const {
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);  // 64MB 청크
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    uint32_t read_u32() const {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    // 중요: 64MB 청크 단위로 쓰기 (대용량 파일 최적화)  
    void write_raw(const void * ptr, size_t len) const {
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64*1024*1024);  // 64MB 청크
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~impl() {
        if (fp) {
            std::fclose(fp);
        }
    }
#else
    // Linux/Unix용 구현: POSIX 표준 함수 사용
    impl(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
// TODO: this ifdef is never true?
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);  // 현재 파일 위치 반환
#endif
        if (ret == -1) {
            throw std::runtime_error(format("ftell error: %s", strerror(errno)));
        }

        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
// TODO: this ifdef is never true?
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);  // 파일 위치 이동
#endif
        if (ret != 0) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    // Linux/Unix에서는 fread/fwrite 사용 (단순함)
    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);  // 전체 길이를 한번에 읽기
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);  // 전체 길이를 한번에 쓰기
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~impl() {
        if (fp) {
            std::fclose(fp);
        }
    }
#endif

    FILE * fp;
    size_t size;
};

llama_file::llama_file(const char * fname, const char * mode) : pimpl(std::make_unique<impl>(fname, mode)) {}
llama_file::~llama_file() = default;

size_t llama_file::tell() const { return pimpl->tell(); }
size_t llama_file::size() const { return pimpl->size; }

int llama_file::file_id() const {
#ifdef _WIN32
    return _fileno(pimpl->fp);
#else
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void * ptr, size_t len) const { pimpl->read_raw(ptr, len); }

uint32_t llama_file::read_u32() const { return pimpl->read_u32(); }

void llama_file::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file::write_u32(uint32_t val) const { pimpl->write_u32(val); }

// ============================================================================
// llama_mmap: 모델 파일 메모리 매핑 시스템
// ============================================================================
/*
 * 핵심 목적: 대용량 모델 파일을 효율적으로 메모리에 매핑
 * 
 * 사용 시나리오:
 * 1. 모델 로딩 시 llama_model_loader에서 사용
 * 2. 몇 GB~수십 GB 크기의 .gguf 파일을 가상 메모리에 매핑
 * 3. 텐서 데이터에 직접 포인터 접근으로 복사 비용 절약
 * 4. 운영체제의 페이지 캐싱 및 prefetch 활용
 * 
 * 장점:
 * - 전체 파일을 RAM에 로드하지 않아도 됨
 * - 필요한 부분만 운영체제가 자동으로 로드
 * - 여러 프로세스 간 동일 파일 공유 가능
 * - 메모리 압박 시 운영체제가 자동으로 unmap
 */

struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    // Linux/Unix용 mmap 구현
    std::vector<std::pair<size_t, size_t>> mapped_fragments;  // 매핑된 메모리 영역 추적

    impl(struct llama_file * file, size_t prefetch, bool numa) {
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;  // 다른 프로세스와 공유 가능
        if (numa) { prefetch = 0; }

#ifdef __linux__
        // 순차 접근 최적화 힌트 (모델 로딩은 보통 순차적)
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        // MAP_POPULATE: 즉시 페이지를 메모리에 로드 (prefetch 시)
        if (prefetch) { flags |= MAP_POPULATE; }
#endif

        // 실제 메모리 매핑 수행
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        // Prefetch 요청 시 운영체제에 사전 로딩 요청
        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        
        // NUMA 시스템에서 랜덤 접근 패턴 힌트
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        mapped_fragments.emplace_back(0, file->size());
    }

    // 페이지 경계에 맞춰 범위 조정 (메모리 매핑은 페이지 단위)
    static void align_range(size_t * first, size_t * last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    // 중요: 부분적 메모리 해제 (SSD offloading에서 활용 가능)
    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);  // 페이지 경계 맞춤
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        // 실제 unmap 수행
        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        // 매핑된 영역 정보 업데이트
        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                // 중간 부분이 제거된 경우: 두 개의 영역으로 분할
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                // 뒷부분이 제거된 경우
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                // 앞부분이 제거된 경우
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
                // 전체가 제거된 경우: 추가하지 않음
            } else {
                // 영향받지 않은 경우
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
        // 모든 매핑된 영역 해제
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    // Windows용 메모리 매핑 구현
    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(numa);

        size = file->size();

        HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

        // 파일 매핑 객체 생성
        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        // 실제 메모리에 매핑
        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        // Windows에서 prefetch 지원 (Windows 8+ 필요)
        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T) std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            LLAMA_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        // Windows에서는 부분 해제 미지원
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~impl() {
        if (!UnmapViewOfFile(addr)) {
            LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    // mmap 미지원 플랫폼
    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif

    void * addr;  // 매핑된 메모리 주소
    size_t size;  // 매핑된 크기
};

llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa) : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}
llama_mmap::~llama_mmap() = default;

size_t llama_mmap::size() const { return pimpl->size; }
void * llama_mmap::addr() const { return pimpl->addr; }

void llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mmap::SUPPORTED  = true;
#else
const bool llama_mmap::SUPPORTED  = false;
#endif

// ============================================================================
// llama_mlock: 메모리 잠금 (스와핑 방지)
// ============================================================================
/*
 * 목적: 중요한 메모리 영역을 RAM에 고정하여 스와핑 방지
 * 
 * 사용 시나리오:
 * - 모델 가중치가 스왑 파일로 이동하는 것을 방지
 * - 일정한 성능 보장 (스왑 I/O로 인한 지연 방지)
 * - 실시간 추론이 중요한 환경에서 사용
 * 
 * 주의사항:
 * - 시스템 메모리 제한(RLIMIT_MEMLOCK)에 의존
 * - 너무 많은 메모리를 잠그면 시스템 불안정 가능
 */

struct llama_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    // Linux/Unix용 메모리 잠금
    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);  // 페이지 크기 단위로 잠금
    }

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

        // 메모리 잠금 실패 시 자세한 에러 메시지 제공
#ifdef __APPLE__
#define MLOCK_SUGGESTION \
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
        "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX)
        // visionOS/tvOS dont't support RLIMIT_MEMLOCK
        // Skip resource limit checks on visionOS/tvOS
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }
#endif

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    // Windows용 메모리 잠금
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            // Working Set 크기 증가 시도
            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            size_t increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    // 메모리 잠금 미지원 플랫폼
    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif

    impl() : addr(NULL), size(0), failed_already(false) {}

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    // 점진적으로 메모리 잠금 확장
    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);  // 페이지 크기로 정렬
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    void * addr;
    size_t size;

    bool failed_already;
};

llama_mlock::llama_mlock() : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;

void llama_mlock::init(void * ptr) { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mlock::SUPPORTED = true;
#else
const bool llama_mlock::SUPPORTED = false;
#endif

size_t llama_path_max() {
    return PATH_MAX;
}
