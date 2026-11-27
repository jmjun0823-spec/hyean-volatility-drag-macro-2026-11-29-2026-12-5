# hyean-volatility-drag-macro-2026-12-w1

**2026년 12월 1주차(11월 29일 ~ 12월 5일)** 주간 증시 전망 및 이벤트 구간을 대상으로, 레버리지 ETF(1x/2x/3x)의 **변동성 드래그(음의 복리 효과)**, 귀무가설($H_0$) 검정, 수급·괴리율 및 슬리피지 리스크를 정량 검증하는 Python·R 파이프라인 저장소입니다[cite: 1, 2, 4].

> 본 저장소의 코드와 리포트는 정량 분석 모델 구현 및 가설 검증을 목적으로 하며, 특정 금융상품의 매수·매도를 권유하지 않습니다[cite: 1].

---

## 1. 분석 대상 및 검증 범위 (`2026-11-29 ~ 2026-12-05`)

* **매크로 및 수급 이벤트 구간**
  * `2026-11-30`: 월말 패시브 리밸런싱 및 연말 포트폴리오 정리·북클로징/배당 시즌 수급 교차[cite: 4]
  * `2026-12-01`: 한국 11월 수출입동향 & 미국 11월 ISM 제조업 PMI[cite: 4]
  * `2026-12-02`: 한국 11월 CPI & 미국 11월 ADP 민간고용[cite: 4]
  * `2026-12-03`: 미국 11월 ISM 서비스업 PMI & 주간 신규 실업수당 청구건수[cite: 4]
  * `2026-12-04`: 미국 11월 고용보고서(비농업 신규고용·실업률) 발표에 따른 변동성·금리·달러 분수령[cite: 4]
* **변동성 드래그 및 거래 리스크 입증**
  * `+2%` 상승 후 `-2%` 하락 교차 등락 시 기초지수(`100 -> 102 -> 99.96`, `-0.04%`), 2배(`99.84`, `-0.16%`), 3배(`99.64`, `-0.36%`) 누적 수익률 검증[cite: 4]
  * 등락폭(`1%`, `2%`, `5%`) 확대 및 보유 기간(`2일`, `10일`, `30일`) 증가에 따른 드래그 단조 심화 검증[cite: 4]
  * 누적 레버리지 수익률이 기초지수 수익률의 $k$배와 같다는 귀무가설($H_0$)에 대한 $t$-검정·윌콕슨 검정 및 고변동성 레짐 비교[cite: 4]
  * 이벤트 일자별 변동성 배수, NAV 대비 괴리율, 호가 스프레드·수급 불균형·종가 리밸런싱을 반영한 예상 슬리피지(`bps`) 산출[cite: 4][cite: 1, 2, 4]

---

## 2. 리포지토리 구조

```text
hyean-volatility-drag-macro-2026-12-w1/
├── .github/
│   └── workflows/
│       └── ci.yml                  # pytest 및 엔드투엔드 워크플로우 CI
├── config/
│   └── universe_config.py          # frozen dataclass 중앙 설정 및 이벤트 일정
├── data/
│   └── sample_macro_fundamentals.csv
├── r_analytics/
│   └── volatility_drag_models.R    # R 회귀 및 가설 교차 검증 스크립트
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # 데이터 로딩, 시뮬레이션/폴백 패널 생성, 캐싱, raw_override_df 지원
│   ├── market_data_fetcher.py      # yfinance 및 종가/거래량 데이터 변환
│   ├── volatility_drag_engine.py   # 변동성 드래그·귀무가설·괴리율/슬리피지 검증 엔진
│   ├── report_builder.py           # 블로그 요약 마크다운 리포트 생성
│   ├── run_live_verification.py    # 실측 데이터 기반 단건 검증 실행기
│   └── pipeline_workflow.py        # 전체 검증 워크플로우 오케스트레이터
├── tests/
│   └── test_volatility_pipeline.py # 단위 및 통합 검증 테스트
├── pyproject.toml
├── requirements.txt
└── README.md
```[cite: 1, 2, 4]

---

## 3. 설치 및 실행 방법

### 환경 설치

```bash
pip install -r requirements.txt
# 실측 시장 데이터 직접 조회 시:
pip install yfinance
