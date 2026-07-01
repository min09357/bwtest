# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

메모리 **read 대역폭(GB/s)** 측정 도구 모음입니다. 1GB hugepage 영역에 64B(캐시라인) 단위 read를 수행하며, 무작위 접근(`randread_bw`)과 순차 접근(`stream_bw`)을 동일한 하네스로 비교합니다. 사용자 대상 상세 문서는 [README.md](README.md)에 있으니, 사용법·인자·출력 해석은 그쪽을 참고하세요. 이 문서는 여러 파일을 함께 읽어야 파악되는 "큰 그림"에 집중합니다.

## 자주 쓰는 명령

```bash
make config                              # config.py를 config_template.py에서 (재)생성 — 최초 1회. 기존 파일 덮어씀
make                                     # access_masks.h 자동 생성 후 두 바이너리 빌드 (config.py 없으면 안내하며 실패)
make WIDTH=256                           # SIMD 로드 폭 강제 (256=AVX2, 64=scalar)
make WIDTH=512 EXTRA_CXXFLAGS=-mavx512f  # native에 없는 폭 강제 시 arch 플래그 동반 필수
make clean                               # 바이너리 + access_masks.h 삭제 (config.py는 보존)

python3 sweep_bw.py            # 코어 1→CORE_MAX 스윕, 무작위 접근 (기본)
python3 sweep_bw.py stream    # 순차 접근 스윕

# 바이너리 직접 실행 (numactl로 코어·노드 고정 권장)
numactl -C 0,2,4 -m 0 ./randread_bw <ncores> <iters_per_thread> <hugepages_1gb> <lines_per_access> <mode>
ACCESS_MAP=<name> ./randread_bw ... 1    # mode=1에서 DRAM 매핑 런타임 선택 (미지정 시 config.ADDR_MAP)
./randread_bw --simd          # 컴파일된 SIMD 폭만 출력하고 종료 (측정·할당 없음)

python3 gen_access_masks.py -o access_masks.h   # mode=1용 헤더 수동 재생성 (모든 SYSTEMS 방출)
```

요구 환경: g++ 13+, Linux(1GB hugepage 지원). 테스트 스위트는 없습니다 — 검증은 실제 측정 실행과 `gen_access_masks.py`의 `self_check`(마스크 정확성 자가검증)로 이뤄집니다.

**실행 전제**: `grep HugePages /proc/meminfo`로 `HugePages_Free >= HUGEPAGES_1GB` 확인. 부족하면 `sudo sh -c 'echo N > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages'`.

## 아키텍처

### 코드 생성 파이프라인 (핵심)

`mode=1`(samebank) 접근은 Python이 계산한 DRAM XOR 마스크를 C++ 바이너리에 컴파일해 넣는 구조입니다. **모든 SYSTEMS의 마스크가 테이블(`ACCESS_MAPS[]`)로 함께 컴파일**되고, 실제로 쓸 매핑은 실행 시 `ACCESS_MAP` 환경변수로 선택합니다(미지정 시 `config.ADDR_MAP`이 컴파일 타임 기본값 `ACCESS_DEFAULT_MAP`). 즉 **매핑을 바꿔도 재컴파일 불필요**(런타임 파일 로딩 없이 선택만). 빌드 의존 체인:

```
config_template.py ──(make config, 명시적 부트스트랩)──▶ config.py [git 미추적]
                                                  │ ADDR_MAP = 기본 매핑
                                                  ▼
address_mapping.py ──(gen_access_masks.py)──▶ access_masks.h ──▶ randread_bw
   (SYSTEMS 레지스트리 +               (ACCESS_MAPS[] 테이블 +      (ACCESS_MAP env로
    GF(2) col_step_masks 솔버)          ACCESS_DEFAULT_MAP)          런타임 매핑 선택)
```

- **`config.py`는 git 미추적**이며 `make config`로 `config_template.py`에서 (재)생성합니다(**덮어쓰기** — 빌드 `all`과 분리된 명시적 단계). config.py 없이 `make` 하면 조용히 만들지 않고 안내하며 실패합니다. 로컬 튜닝은 `config.py`를, 커밋할 프로젝트 기본값은 `config_template.py`를 수정하세요. `sweep_bw.py`·`gen_access_masks.py`가 모두 `import config`로 읽습니다.
- `config.py`나 `address_mapping.py`를 수정하면 다음 `make`에서 `access_masks.h`가 재생성됩니다(Makefile 의존성에 명시). `config.ADDR_MAP` 변경은 **기본값**만 바꾸므로 재빌드가 필요하지만, 이미 컴파일된 다른 매핑은 `ACCESS_MAP` env로 재빌드 없이 즉시 사용 가능합니다.

### samebank 모드와 GF(2) 솔버 (mode=1)

`lines_per_access`개 캐시라인을 **같은 channel/rank/bank-group/bank·같은 row 안의 인접 column**에 배치하는 row-buffer-hit 패턴입니다. 실제 메모리 컨트롤러는 column을 결정하는 물리 비트가 bank 선택 XOR 함수의 입력과 겹치므로, column 비트만 뒤집으면 bank가 바뀝니다.

- [address_mapping.py](address_mapping.py): `SYSTEMS`에 시스템별 주소 매핑(offset/channel/rank/bank_group/bank/row/column 비트마스크)을 등록. `col_step_masks()`가 GF(2) XOR basis(`_XorBasis`)로, column 비트 하나만 토글하면서 bank/row는 보존하는 최소 마스크를 계산. 새 시스템은 여기 `SYSTEMS`에 추가.
- [gen_access_masks.py](gen_access_masks.py): **모든 SYSTEMS**를 순회하며 솔버로 마스크를 뽑고 `address_mapping.decode()`로 256개 무작위 주소에 대해 자가검증(`self_check`) 후 `access_masks.h`를 방출 — `struct AccessMap { name, num_col_bits, col_step_mask[ACCESS_MAX_COL_BITS] }`의 `ACCESS_MAPS[]` 테이블 + `ACCESS_DEFAULT_MAP`(=`config.ADDR_MAP`). `ACCESS_MAX_COL_BITS`는 `MAX_COL_BITS`(=4, `lines_per_access ≤ 16 = 2⁴`이므로)에서 방출.
- [randread_bw.cpp](randread_bw.cpp): `main`에서 mode==1이면 `ACCESS_MAP` env(미지정 시 `ACCESS_DEFAULT_MAP`)로 `ACCESS_MAPS[]`에서 항목(`sel`)을 고르고(없으면 사용 가능 목록 나열 후 에러), `sel->col_step_mask`를 조합해 `col_masks[d]`(column을 d만큼 이동하는 마스크)를 만든 뒤 hot loop에서 `off ^ col_masks[k]`로 접근. `lines_per_access > 2**sel->num_col_bits`면 런타임 에러. 매핑 선택·검증·col_masks 생성은 모두 측정 루프 **밖**이라 hot path 성능 영향 없음.

### 컴파일 타임 SIMD 폭 선택

[bw_width.h](bw_width.h)가 SIMD 로드 폭(512/256/64)을 **전처리기 `#if`로** 결정합니다(런타임 분기 아님 — 선택 안 된 코드는 바이너리에 미생성). 두 `.cpp`가 공유하며 각자 hot loop를 `#if BW_SIMD_WIDTH`로 분기. 우선순위: `-DBW_SIMD_WIDTH` 사용자 지정 > `-march=native`의 ISA 자동 감지(`__AVX512F__`→512, `__AVX2__`→256, 없으면 64 스칼라). 지원 안 되는 폭 강제 시 `#error`로 실패.

**설계 원칙**: 64B 캐시라인 1개 = DRAM 트랜잭션 1개. 라인당 로드 1회를 유지하면 폭과 무관하게 미스 시 64B 전체가 올라오고, 대역폭은 `반복 × 64B`로 계산하므로 **로드 폭은 측정 정확성에 무관**. Makefile은 `-mavx512f`를 강제하지 않아(바이너리 이식성) `WIDTH=`는 주로 다른 머신·시뮬레이터 연동용.

### 측정 하네스 (두 바이너리 공통 패턴)

- **병렬화**: `std::jthread` + `std::barrier`, 코어당 1스레드. 코어 핀은 `numactl -C/-m`로 **외부에서** 설정(코드 내 affinity 설정 없음).
- **타이밍**: 모든 스레드가 barrier에서 만난 뒤 tid 0이 `g_t_start`(atomic) 기록 → 측정 구간이 스레드 생성/워밍업을 제외.
- **DCE 방지**: 읽은 데이터를 SIMD/스칼라 XOR로 `sink`에 누적, `main`에서 전 스레드 sink를 XOR해 `Checksum`으로 출력. 같은 인자면 checksum이 항상 동일해야 함.
- **워밍업**: `mmap`(`MAP_HUGETLB | MAP_HUGE_1GB`) 후 순차 write로 물리 페이지 커밋. `stream_bw`는 위치 종속 값으로 채워 의미 있는 checksum 확보(`randread_bw`는 `memset` 균일 채움).

### randread_bw 특유 설계

- **주소 생성**: HPCC RandomAccess POLY Galois LFSR. 스레드당 `UNROLL`(16)개 독립 스트림을 `hpcc_starts()` skip-ahead로 초기화해 MLP(다중 미해결 로드) 극대화. 전체 스트림 = `ncores × 16`.
- **LFSR 마스크 주소 방식** 때문에 `hugepages_1gb`는 **반드시 2의 거듭제곱**. `nblk-1`(mode 0) 또는 `ncl-1`(mode 1) 마스크로 인덱싱.
- **템플릿 디스패치**: `thread_func<LINES, MODE>`에서 `LINES`·`MODE`를 컴파일 타임 상수화 → 내부 k-루프가 -O3에서 완전 unroll, hot path에 런타임 분기 없음. `main`의 `switch(lines_per_access)` + `BW_DISPATCH` 매크로가 한 번만 디스패치.
- **트래픽 불변식**: `total_bytes = ncores × iters_per_thread × 64` — `lines_per_access`·`mode`와 **무관하게 고정**. 그래서 같은 `iters`로 granularity/mode만 바꿔 공정 비교 가능. `iters_per_thread`는 `UNROLL × lines_per_access` 배수로 내림 조정됨. `Total accesses`(GUPS 기준)는 `iters / lines_per_access`로 별개.

## 컨벤션

- 문서(CLAUDE.md, README.md 등)는 한글, 코드 주석은 영어로 작성합니다.
- `randread_bw.cpp`와 `stream_bw.cpp`는 하네스 구조(ThreadArg/barrier/sink reduce/워밍업/결과 출력)를 의도적으로 대칭 유지합니다 — 한쪽을 고치면 다른 쪽 대응부도 함께 확인하세요.
