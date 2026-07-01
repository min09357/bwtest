# randread_bw — 메모리 읽기 대역폭 측정 도구 모음

시스템의 메모리 **read bandwidth(GB/s)** 를 측정합니다.
1GB 거대 페이지(hugepage) 영역에 대해 64B(캐시라인) 단위 read를 수행하며,
**무작위 접근**과 **순차 접근** 두 패턴을 동일한 하네스로 비교할 수 있습니다.

| 도구 | 패턴 | 설명 |
|---|---|---|
| `randread_bw` | 무작위 read | HPCC RandomAccess POLY LFSR 주소 생성, 캐시라인 단위 random read |
| `stream_bw` | 순차 read | 스레드별 연속 구간을 캐시라인 stride로 순차 스캔 (STREAM 류) |
| `sweep_bw.py` | — | 1코어부터 N코어까지 스윕하며 대역폭·이론 DRAM 피크 대비 % 출력 |
| `config.py` | — | DIMM·NUMA·hugepage·스윕 범위 등 모든 설정을 관리하는 설정 파일 (git 미추적, `make config`로 생성) |
| `config_template.py` | — | `config.py`의 기본값 템플릿 (git 추적) |
| `address_mapping.py` | — | 시스템별 DRAM 주소 매핑(채널/랭크/뱅크그룹/뱅크/row/col) 레지스트리 + XOR 마스크 솔버 |
| `gen_access_masks.py` | — | `address_mapping.SYSTEMS` 전체를 `access_masks.h` 테이블로 생성 (`make` 시 자동 실행) |

## 설계 요점

| 항목 | 결정 |
|---|---|
| 메모리 영역 | `HUGEPAGES_1GB` × 1GB (`MAP_HUGE_1GB`), 연속 가상 주소 (기본 4GB) |
| 접근 단위 | 1~16 캐시라인 (블록 정렬; `lines_per_access` 파라미터, 기본 1). SIMD 로드 폭은 컴파일 타임 선택 |
| 연산 종류 | Read 전용 |
| 병렬화 | `std::jthread` + `std::barrier`, 코어당 1스레드 |
| 코어 핀 | `numactl -C <cpulist> -m <node>` 로 외부에서 설정 (`sweep_bw.py` 가 자동 처리) |
| DCE 방지 | 읽은 데이터를 SIMD XOR 또는 스칼라 XOR로 누적 → checksum 출력 |
| SIMD 폭 | 컴파일 타임 자동 감지 (AVX-512→512, AVX2→256, 없음→64비트 스칼라); `make WIDTH=N`으로 강제 지정 가능 |

### randread_bw 전용

| 항목 | 결정 |
|---|---|
| 주소 생성 | HPCC POLY Galois LFSR, 스트림별 skip-ahead(`hpcc_starts`) |
| 스트림 수 | `ncores × 16` (스레드당 16개 독립 스트림 → MLP 극대화) |
| 접근 granularity | `lines_per_access` 캐시라인을 한 번에 fetch (`mode`로 배치 방식 선택, 아래 참고) |
| 다중 라인 unroll | `thread_func<LINES, MODE>` 템플릿으로 N·MODE를 컴파일 타임 상수화; 내부 k-루프가 -O3에서 완전 unroll되어 런타임 분기 없음. `main`에서 `switch`+분기로 한 번만 디스패치 |
| 총 트래픽 | `iters_per_thread × 64 B` (N·mode 무관 고정) — 같은 iters로 granularity·mode만 바꾸면 공정 비교 가능 |

#### 접근 방식 (mode)

`lines_per_access`개의 캐시라인을 어디에 배치할지를 `mode`(정수 0/1, 기본 0)로 선택합니다.

| mode | 이름 | 설명 |
|---|---|---|
| 0 | `consecutive` (기본) | 랜덤으로 고른 `lines_per_access × 64B` 정렬 블록에서 연속 캐시라인 접근 (`addr, addr+64, addr+128, …`). 기존 동작과 동일 |
| 1 | `samebank` | 랜덤으로 고른 캐시라인 주소를 `access_masks.h`의 마스크와 XOR 하여, 모든 캐시라인이 **같은 channel/rank/bank-group/bank(뱅크)·같은 row**에서 **인접한 column**만 바뀌도록 접근 (row buffer hit 패턴) |

`mode=1`은 `address_mapping.py`에 정의된 DRAM 주소 매핑을 기반으로 XOR 마스크를 계산합니다. 주소 매핑에서 column을 결정하는 물리 비트가 channel/rank/bank-group/bank를 결정하는 XOR 함수의 입력과 겹치는 경우가 많아, 단순히 column 비트만 뒤집으면 뱅크가 바뀌어 버립니다. `address_mapping.col_step_masks()`는 GF(2) 선형대수(XOR basis)로 이 겹침을 상쇄하는 최소 마스크를 계산합니다 — 자세한 내용은 `address_mapping.py`의 모듈 docstring과 `gen_access_masks.py` 참고.

**매핑은 런타임에 선택**합니다. `access_masks.h`에는 `address_mapping.SYSTEMS`의 **모든** 매핑이 테이블로 컴파일되며, 실행 시 `ACCESS_MAP=<name>` 환경변수로 고릅니다. 미지정 시 `config.ADDR_MAP`(컴파일 타임 기본값)을 사용합니다. 따라서 매핑을 바꿔도 재컴파일이 필요 없습니다.

```bash
ACCESS_MAP=arrow_1ch_1dpc_2rank_32gb numactl -C 0 -m 0 ./randread_bw 4 100000000 4 4 1
```

없는 이름을 주면 사용 가능한 매핑 목록을 출력하고 종료합니다. `mode=1`이 지원하는 최대 `lines_per_access`는 선택된 매핑의 column 폭(`num_col_bits`)에 따라 결정되며, 이를 초과하면 실행 시 에러로 종료합니다.

#### 주소 충돌에 대하여
스트림마다 64비트 `ran` 수열 구간은 겹치지 않지만, 하위 비트 인덱스가 같아 **캐시라인 주소가 코어 간·반복 간 겹칠 수 있습니다.**
이는 의도된 동작입니다. read 전용이라 coherence 비용이 없고, 작업집합이 일반적으로 L3(64MB)를 크게 초과하여 재방문해도 대부분 DRAM 접근으로 이어져 측정이 유효합니다.

### stream_bw 전용
영역을 코어 수만큼 연속 구간으로 분할하고, 각 스레드가 자기 구간을 캐시라인 stride로 순차 스캔합니다. `iters_per_thread`만큼 채울 때까지 구간을 wrap-around 반복합니다. 순차 접근이라 프리페처·row buffer 효과가 모두 살아 있어 `randread_bw`보다 훨씬 높은 대역폭이 나오는 것이 정상입니다.

## 빌드

```bash
make config     # config.py를 config_template.py에서 (재)생성 — 최초 1회 (기존 파일 덮어씀)
make            # access_masks.h 자동 생성 후 randread_bw, stream_bw 빌드 (SIMD 폭 자동 감지)
make WIDTH=256  # AVX2 256비트로 강제
make WIDTH=64   # 스칼라(64비트)로 강제
make WIDTH=512 EXTRA_CXXFLAGS=-mavx512f  # AVX-512 강제 (해당 ISA 지원 머신/시뮬레이터용)
```

요구 환경: g++ 13+, Linux(1GB hugepage 지원).

`config.py`는 사용자 입력이라 빌드와 분리돼 있습니다 — 최초에 `make config`로 만든 뒤 편집하세요. config.py 없이 `make`를 실행하면 조용히 만들지 않고 안내 메시지와 함께 실패합니다.

`access_masks.h`는 `address_mapping.SYSTEMS` 전체를 `gen_access_masks.py`가 테이블로 생성하는 파일로, `randread_bw`의 `mode=1`(samebank) 접근에 사용됩니다(실행 시 `ACCESS_MAP`으로 매핑 선택, 기본값은 `config.ADDR_MAP`). `config.py`나 `address_mapping.py`를 수정하면 다음 `make` 시 재생성됩니다. 수동 실행: `python3 gen_access_masks.py [-o access_masks.h]`.

## SIMD 로드 폭 (512 / 256 / 64)

64B 캐시라인 1개 = DRAM 트랜잭션 1개입니다. 라인을 한 번만 건드리면(폭과 무관) 미스 시 64B 전체가 올라오고, 대역폭 계산은 `반복 횟수 × 64B`로 하므로 **로드 폭은 측정 정확성에 영향을 주지 않습니다.** 따라서 라인당 1 로드를 유지하면서 CPU가 지원하는 폭만 맞춰주면 됩니다.

폭은 **컴파일 타임에 결정**됩니다 (런타임 `if` 분기가 아니라 `#if` 전처리기 분기 — 선택되지 않은 코드는 바이너리에 아예 생성되지 않음). 폭 결정 로직은 [bw_width.h](bw_width.h)에 모여 있습니다.

| 빌드 명령 | 결과 폭 | 비고 |
|---|---|---|
| `make` | 자동 감지 | `-march=native` 기준: AVX-512→512, AVX2→256, 둘 다 없으면 64비트 스칼라 |
| `make WIDTH=256` | 256 (AVX2) | 명시적 강제 |
| `make WIDTH=64` | 64 (scalar) | SIMD ISA 불필요 (어느 CPU/시뮬레이터든 동작) |
| `make WIDTH=512 EXTRA_CXXFLAGS=-mavx512f` | 512 (AVX-512) | native에 없는 폭 강제 시 arch 플래그 동반 필요 |

> `WIDTH=`는 빌드 호스트가 아닌 **다른 머신·시뮬레이터에 바이너리를 연동**할 때 폭을 명시 고정하기 위한 옵션입니다. 미지정 시 빌드 호스트 ISA에 맞춰 자동 감지됩니다. native에 없는 폭을 `WIDTH=`로만 강제하면 컴파일 시 `#error`로 깔끔히 실패합니다.

선택된 폭은 세 곳에서 확인할 수 있습니다:

- **컴파일 시** — 빌드 로그: `note: '#pragma message: BW_SIMD_WIDTH = 256 (AVX2)'`
- **실행 시** — 시작 줄: `... region=4 GB  lines/access=1  simd=256 (AVX2)`
- **조회 전용** — `./randread_bw --simd` / `./stream_bw --simd` → 폭 한 줄만 출력하고 즉시 종료 (측정/메모리 할당 없음; `sweep_bw.py`가 헤더 표시에 사용)

## 실행

### 개별 벤치마크

```bash
# 무작위 read (기본값: 16코어, 1억회, 4GB 영역, 라인 1개/접근, mode=0)
./randread_bw [ncores] [iters_per_thread] [hugepages_1gb] [lines_per_access] [mode]

# 순차 read (동일 인터페이스, lines_per_access/mode 없음)
./stream_bw [ncores] [iters_per_thread] [hugepages_1gb]

# 예시: 8코어, 5천만회, 4GB 영역, 4라인/접근, 연속 접근(mode=0)
numactl -C 0,2,4,6,8,10,12,14 -m 0 ./randread_bw 8 50000000 4 4 0
# 동일 조건, 같은 뱅크/row 내 인접 column 접근(mode=1) — 기본 매핑(config.ADDR_MAP)
numactl -C 0,2,4,6,8,10,12,14 -m 0 ./randread_bw 8 50000000 4 4 1
# mode=1에서 매핑을 런타임 지정 (재컴파일 불필요)
ACCESS_MAP=arrow_1ch_1dpc_2rank_32gb numactl -C 0,2,4,6,8,10,12,14 -m 0 ./randread_bw 8 50000000 4 4 1
numactl -C 0,2,4,6,8,10,12,14 -m 0 ./stream_bw   8 50000000 4
```

- `ncores`: 사용할 코어 수 (기본 16).
- `iters_per_thread`: 스레드당 처리할 총 64B 캐시라인 수 (기본 1억). `randread_bw`는 `UNROLL(16) × lines_per_access` 배수로 내림 조정됨. 총 트래픽은 `iters × 64B`로 `lines_per_access`/`mode`와 무관하게 고정.
- `hugepages_1gb`: 할당할 1GB hugepage 수 (기본 4). `randread_bw`는 **반드시 2의 거듭제곱**(1, 2, 4, 8, …)이어야 함 (LFSR 마스크 주소 방식 제약). `stream_bw`는 임의의 양수 가능.
- `lines_per_access` (`randread_bw` 전용, 기본 1): 한 번의 무작위 접근에서 fetch할 캐시라인 수. 허용값 `{1, 2, 4, 8, 16}`. 접근 횟수 = `iters / lines_per_access`.
- `mode` (`randread_bw` 전용, 기본 0): `lines_per_access`개 캐시라인의 배치 방식.
  - `0` = consecutive: `lines_per_access × 64B`에 정렬된 블록에서 연속 캐시라인 (`addr, addr+64, …`).
  - `1` = samebank: 같은 channel/rank/bank-group/bank·같은 row 안의 인접 column을 XOR 마스크로 접근 (row buffer hit 패턴). `config.ADDR_MAP`으로 선택한 매핑이 지원하는 최대 `lines_per_access`를 넘으면 에러.
- **주의**: CPU 번호가 NUMA 노드별로 인터리브된 시스템(예: node0=짝수, node1=홀수)에서 numactl 없이 실행하면 스레드가 여러 노드에 걸쳐 스케줄될 수 있어 측정이 오염됩니다. `sweep_bw.py`는 항상 numactl로 감싸 실행합니다.

### 코어 스윕 (sweep_bw.py)

1코어부터 `CORE_MAX`까지 코어 수를 늘려가며 대역폭과 **이론 DRAM 피크 대비 %** 를 막대그래프로 출력합니다.

```bash
python3 sweep_bw.py          # 무작위 접근 (기본)
python3 sweep_bw.py rand     # 무작위 접근
python3 sweep_bw.py stream   # 순차 접근
```

`config.py`에서 모든 설정을 관리합니다. 스크립트를 실행하기 전에 이 파일을 편집하세요.
`config.py`는 git에 추적되지 않는 로컬 파일로, `make config`로 `config_template.py`에서 (재)생성합니다 (**덮어쓰기** — 빌드와 분리된 명시적 부트스트랩 단계).

| 항목 | 상수 | 의미 |
|---|---|---|
| DIMM | `DIMM_TRANSFER_RATE_MT_S` | DIMM 전송률 (예: DDR5-5200 → 5200) |
| DIMM | `DIMM_CHANNELS` | 채워진 메모리 채널 수 |
| NUMA | `NUMA_NODE` | 측정에 사용할 NUMA 노드 번호 (CPU affinity `-C` 및 메모리 `-m` 모두 이 노드로 고정) |
| Hugepage | `HUGEPAGES_1GB` | 할당할 1GB hugepage 수 (기본 4). rand 모드는 2의 거듭제곱 필수 |
| 스윕 | `CORE_START` / `CORE_MAX` / `CORE_STEP` | 스윕 코어 범위·증가폭 |
| 스윕 | `ITERS_PER_THREAD` | 스레드당 처리할 총 캐시라인 수 (총 트래픽 = iters × 64B, N 무관) |
| rand | `LINES_PER_ACCESS` | 접근당 캐시라인 수 `{1,2,4,8,16}` (rand 모드 전용, 기본 1) |
| rand | `ACCESS_MODE` | 접근 방식: `0`=consecutive(기본), `1`=samebank (rand 모드 전용) |
| rand | `ADDR_MAP` | `ACCESS_MODE=1`의 **기본** DRAM 주소 매핑 — `address_mapping.SYSTEMS`의 키. 실행 시 `ACCESS_MAP` env로 덮어쓸 수 있음(`sweep_bw.py`는 이 값을 자동 전달) |
| 기타 | `USE_SUDO` | 1GB hugepage 할당에 sudo가 필요하면 `True` |

이론 피크는 `전송률(MT/s) × 8 B × 채널 수 / 1000` 으로 계산합니다.
`sudo dmidecode -t memory | grep -E "Speed|Configured"` 로 실제 DIMM 값을 확인해 채우세요.

## 출력 해석

```
ncores=16  iters/thread=100000000  streams=256  region=4 GB  lines/access=1 (64 B)  mode=0 (consecutive)  simd=256 (AVX2)
Warming up 4 GB ... done
```

`sweep_bw.py` 헤더 출력 예시:

```
=================================================================
 randread_bw — core sweep  [random access]
=================================================================
  DIMM rate   : 4800 MT/s × 8 B × 1 ch  →  peak 38.4 GB/s
  NUMA node   : 0  (CPUs: 0,1,2,3,...)
  Core range  : 1 .. 32  (step 1)
  Iters/thread: 100,000,000
  Hugepages   : 4 × 1GB  (4 GB region)
  Lines/access: 1  (64 B/access)
  Access mode : 0 (consecutive)
  Binary      : /path/to/randread_bw
  SIMD width  : 256 (AVX2)
=================================================================
```

바이너리 직접 실행 결과 예시:

```

=== Results ===
Elapsed        : 4.123 s
Total accesses : 1.600e+09  (1.024e+11 bytes)
Bandwidth      : 24.840 GB/s  (23.133 GiB/s)
GUPS           : 0.3880
Checksum       : 3f2a1b4c8d9e0f12
```

- **Bandwidth**: read 대역폭. `randread_bw`(무작위)는 `stream_bw`(순차)보다 훨씬 낮은 것이 정상 — 캐시라인 단위 무작위라 row buffer / 프리페처 효과가 없습니다.
- **Total accesses**: 무작위 주소 생성 횟수 (= `iters_per_thread / lines_per_access × ncores`). 총 바이트는 이것과 무관하게 `iters × 64B` 고정.
- **GUPS**: 참고용. HPCC 정의와 달리 여기서는 read-only 기준이며, 주소 생성 횟수 기준.
- **Checksum**: DCE(dead-code elimination) 방지용 누적값. 같은 인자면 항상 동일해야 함.

## 주의사항

- 실행 전 `grep HugePages /proc/meminfo` 로 `HugePages_Free >= HUGEPAGES_1GB` 확인.
  부족하면 `sudo sh -c 'echo N > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages'` 로 할당.
- `randread_bw`의 `hugepages_1gb`는 **2의 거듭제곱**만 허용 (1, 2, 4, 8, …). 그 외 값은 에러로 종료.
- `lines_per_access`는 `{1, 2, 4, 8, 16}` 중 하나만 허용. 그 외 값은 에러로 종료.
- `mode=1`(samebank)은 `lines_per_access`가 선택된 매핑(`ACCESS_MAP` env, 미지정 시 `config.ADDR_MAP`)이 지원하는 최대값(`num_col_bits`로 결정)을 넘으면 에러로 종료. 없는 매핑 이름을 주면 사용 가능한 목록을 출력하고 종료.
- 16GB 단일 채널 구성이면 DRAM 대역폭 상한이 낮아질 수 있음.
  `sudo dmidecode -t memory` 로 DIMM 수·속도 확인 권장.
