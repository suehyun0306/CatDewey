import streamlit as st
import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import os
import platform
import re # 정규표현식 추가

# --------------------------------------------------------------------------------
# 1. 기본 설정 및 한글 폰트
# --------------------------------------------------------------------------------
st.set_page_config(page_title="도서관 데이터 분석 챗봇", layout="wide")

def set_korean_font():
    system_name = platform.system()
    if system_name == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    elif system_name == 'Windows': # Windows
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Linux': # Linux (Streamlit Cloud 등)
        # 나눔고딕 등이 설치되어 있다고 가정하거나, 없으면 기본 폰트 사용 후 경고
        # 실제 배포 시에는 packages.txt에 fonts-nanum을 추가해야 함
        try:
            plt.rc('font', family='NanumGothic')
        except:
            pass # 폰트가 없으면 기본 폰트 사용 (깨질 수 있음)
    plt.rc('axes', unicode_minus=False)

set_korean_font()

# --------------------------------------------------------------------------------
# 2. 사용자 정의 스키마 정보 (LLM 제공용)
# --------------------------------------------------------------------------------
FIXED_SCHEMA_INFO = """
[데이터베이스 스키마]

1) base_info (도서관 기본 정보)
  - 도서관코드 (INTEGER, PK)
  - 도서관명 (TEXT)
  - 구분 (TEXT)
  - 시도 (TEXT, FK → pop.(시도, 시군구))
  - 시군구 (TEXT, FK → pop.(시도, 시군구))

2) holding (장서 현황)
  - 도서관코드 (INTEGER, PK, FK → base_info.도서관코드)
  - 총장서 (INTEGER)
  - 국외서 (INTEGER)

3) fac (시설 현황)
  - 도서관코드 (INTEGER, PK, FK → base_info.도서관코드)
  - 면적_도서관 부지 면적 (FLOAT)
  - 면적_도서관 건물 연면적 (FLOAT)
  - 면적_도서관 서비스 제공 면적 (FLOAT)
  - 좌석수_총 좌석수 (INTEGER)
  - 좌석수_어린이 열람석 (INTEGER)
  - 좌석수_노인 및 장애인 열람석 (INTEGER)

4) user (이용자 현황)
  - 도서관코드 (INTEGER, PK, FK → base_info.도서관코드)
  - 회원_어린이 (INTEGER)
  - 회원_청소년 (INTEGER)
  - 회원_성인 (INTEGER)
  - 방문자수 (INTEGER)

5) service (서비스 현황)
  - 도서관코드 (INTEGER, PK, FK → base_info.도서관코드)
  - 취약계층서비스이용수_합계 (INTEGER)
  - 취약계층서비스이용수_장애인 (INTEGER)
  - 취약계층서비스이용수_노인 (INTEGER)
  - 취약계층서비스이용수_다문화 (INTEGER)
  - 취약계층관련예산_합계 (INTEGER)
  - 취약계층관련예산_장애인 (INTEGER)
  - 취약계층관련예산_노인 (INTEGER)
  - 취약계층관련예산_다문화 (INTEGER)
  - 취약계층공간_장애인 (TEXT)
  - 취약계층공간_노인 (TEXT)
  - 취약계층공간_다문화 (TEXT)
  - 어린이실 (TEXT)
  - 어린이서비스_이용수 (INTEGER)
  - 어린이자료_인쇄수 (INTEGER)

  6) pop (지역 인구 마스터)
  - 시도 (TEXT, PK → 복합키의 일부)
  - 시군구 (TEXT, PK → 복합키의 일부)
  - 총인구 (INTEGER)
  - 어린이인구 (INTEGER)
  - 노인인구 (INTEGER)
  - 장애인인구 (INTEGER)
  - 다문화인구 (INTEGER)
"""

# --------------------------------------------------------------------------------
# 3. 데이터 적재 (ETL) 로직
# --------------------------------------------------------------------------------
DB_PATH = 'CatDewey.db'

def read_csv_robust(file_path):
    encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    # 최후의 수단
    return pd.read_csv(file_path, encoding='cp949', errors='replace')

def initialize_database():
    csv_files = {
        'base_info': 'T1_도서관기본정보.csv',
        'holding': 'T2_장서정보.csv',
        'fac': 'T3_시설현황.csv',
        'user': 'T4_이용자정보_지역포함.csv',
        'service': 'T5_지식정보취약계층서비스.csv',
        'pop': 'T6_지역인구_시군구포함.csv'
    }
    
    # 파일 존재 여부 확인 (디버깅 편의를 위해 없는 파일은 건너뛰고 진행)
    existing_files = {k: v for k, v in csv_files.items() if os.path.exists(v)}
    
    if not existing_files:
        # DB가 이미 있다면 굳이 에러를 띄우지 않고 기존 DB 사용
        if os.path.exists(DB_PATH):
            return True
        st.error(f"❌ CSV 파일이 작업 폴더에 없습니다. 다음 파일들을 확인해주세요: {list(csv_files.values())}")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        progress_bar = st.progress(0)
        
        total = len(existing_files)
        for i, (table, path) in enumerate(existing_files.items()):
            df = read_csv_robust(path)
            df.to_sql(table, conn, if_exists='replace', index=False)
            progress_bar.progress((i + 1) / total)

        conn.commit()
        conn.close()
        progress_bar.empty()
        st.toast("데이터베이스가 성공적으로 업데이트되었습니다!", icon="✅")
        return True
    except Exception as e:
        st.error(f"DB 생성 오류: {e}")
        return False

# --------------------------------------------------------------------------------
# 4. LLM 및 분석 함수
# --------------------------------------------------------------------------------

def nl_to_sql(client, question):
    system_prompt = f"""
    당신은 SQLite 전문가입니다. 아래 스키마를 보고 질문을 SQL로 변환하세요.

    {FIXED_SCHEMA_INFO}

    [규칙]
    1. 여러 문장을 세미콜론으로 이어서 쓰지 마세요. 그러나 제일 마지막에는 세미콜론을 하나 붙이세요.
    2. INSERT, UPDATE, DELETE 등 데이터 변경 구문은 절대 사용하지 마세요. (읽기 전용)
    3. 존재하지 않는 테이블이나 컬럼 이름을 지어내지 말고, 위에 정의된 스키마만 사용하세요.
    4. GROUP BY, ORDER BY, LIMIT, JOIN 등을 자유롭게 활용할 수 있습니다.
    5. 결과를 JSON 형식의 문자열로만 출력하세요.
    6. 시군구의 결과를 물어볼 때는 시도와 시군구를 기준으로 GROUP BY 하세요.
    7. 시도의 결과를 물어볼 때는 시도를 기준으로 GROUP BY 하세요.
    8. SUM 등 계산 함수를 적절하게 사용하세요.
    9. SELECT나 WHERE 절에 사용된 컬럼이 있는 테이블은 반드시 FROM이나 JOIN 절에 포함되어야 합니다.
    10.  포괄적 쿼리 제공: 데이터의 집계(SUM, AVG), 비교(JOIN), 필터링(WHERE), 순위 지정(ORDER BY) 등을 자유롭게 활용하여 유의미한 분석 결과를 도출합니다.
    11.  데이터 무결성 유지: 쿼리 작성 시, 테이블 간의 **PK-FK 관계**를 정확히 이해하고 JOIN을 활용하여 데이터의 정합성을 유지합니다.
    12.  스키마 활용: 쿼리 작성 시, 아래에 제시된 테이블 및 칼럼을 정확히 활용해야 합니다.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # [수정된 부분] 안전장치 추가: 'sql' 키가 없으면 찾아내거나 에러 메시지로 대체
        if "sql" not in data:
            if "query" in data:
                # AI가 실수로 'query'라는 키를 쓴 경우 처리
                data["sql"] = data["query"]
            elif "SQL" in data:
                 # AI가 대문자 'SQL'을 쓴 경우 처리
                data["sql"] = data["SQL"]
            else:
                # 어떤 키도 없으면 강제로 에러 SQL 주입 (KeyError 방지)
                data["sql"] = "-- SQL 생성 실패: AI가 올바른 형식을 반환하지 않음"
                if "explanation" not in data:
                    data["explanation"] = f"AI 응답 오류: {content}"
            
        if "explanation" not in data:
            data["explanation"] = "자동 생성된 쿼리입니다."
            
        return data

    except Exception as e:
        # JSON 파싱 실패 등 아예 오류가 난 경우
        return {
            "sql": "-- Error", 
            "explanation": f"쿼리 생성 실패: {str(e)}"
        }

def generate_viz_code(client, df, question):
    # 데이터프레임 정보 요약
    df_head = df.head().to_markdown()
    columns = list(df.columns)
    
    system_prompt = f"""
    당신은 Python 데이터 시각화 전문가입니다.
    Pandas DataFrame `df`가 주어졌습니다.
    컬럼: {columns}
    데이터 예시:
    {df_head}
    
    사용자 질문: "{question}"
    
    [요구사항]
    1.  분석 목표 수용: 사용자 요청의 **분석적 가치**를 극대화하는 시각화와 리포트를 작성합니다. 
    2. 한글 폰트 설정은 이미 되어 있습니다 (plt.rc).
    3. `plt.figure(figsize=(10, 6))` 등으로 그래프 크기를 적절히 설정하세요.
    4. `plt.show()`는 절대 사용하지 마세요.
    5. 오직 실행 가능한 Python 코드만 출력하세요.
    6. 변수명은 `df`를 사용하세요.
    7. 결과 데이터에 중복값이 없도록 하세요. 
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "시각화 코드를 작성해줘."}
            ],
            temperature=0
        )
        code = response.choices[0].message.content
        # 마크다운 제거 정규식
        clean_code = re.sub(r"```python|```", "", code).strip()
        return clean_code
    except Exception as e:
        return f"# 시각화 코드 생성 실패: {e}"

def generate_report(client, df, question):
    summary = df.head(10).to_markdown()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "데이터 분석가로서 결과를 요약하고 인사이트를 제공하세요."},
                {"role": "user", "content": f"질문: {question}\n데이터:\n{summary}"}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception:
        return "리포트 생성에 실패했습니다."

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.divider()
    if st.button("🔄 DB 데이터 초기화/갱신"):
        initialize_database()

st.title("📚 도서관 데이터 분석 AI")

# API 키 확인
if not api_key:
    st.info("👈 사이드바에 OpenAI API Key를 입력하면 시작됩니다.")
    st.stop()

# DB 확인 및 초기화 시도
if not os.path.exists(DB_PATH):
    if not initialize_database():
        st.stop() # DB 생성 실패시 중단

client = OpenAI(api_key=api_key)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 도서관 데이터에 대해 물어보세요."}]
# [중요] 분석 결과를 유지하기 위한 세션 상태
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# 이전 대화 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 마지막 분석 결과가 있다면 다시 보여주기 (Rerun 대응)
if st.session_state.last_result:
    res = st.session_state.last_result
    with st.expander("📊 분석 결과 보기", expanded=True):
        st.code(res['query'], language="sql")
        st.info(res['explanation'])
        
        tab1, tab2, tab3 = st.tabs(["📋 데이터", "📈 시각화", "📝 리포트"])
        with tab1:
            st.dataframe(res['df'])
        with tab2:
            if res['viz_code']:
                # exec 안전 실행
                try:
                    # plt.show를 무력화하여 에러 방지
                    exec_globals = {'pd': pd, 'plt': plt, 'sns': sns, 'st': st}
                    exec_locals = {'df': res['df']}
                    # plt.show가 호출되어도 아무일도 안 일어나게 dummy 함수 할당
                    exec("plt.show = lambda: None", exec_globals) 
                    exec(res['viz_code'], exec_globals, exec_locals)
                    st.pyplot(plt.gcf())
                    plt.clf() # 렌더링 후 초기화
                except Exception as e:
                    st.error(f"시각화 코드 실행 오류: {e}")
                    st.code(res['viz_code'])
        with tab3:
            st.write(res['report'])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지 화면 표시 및 저장
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. AI 답변 처리
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("데이터 분석 중입니다..."):
            # 1) SQL 생성
            res_sql = nl_to_sql(client, prompt)
            query = res_sql['sql']
            explanation = res_sql['explanation']
            
            if "SELECT" not in query.upper():
                st.error("올바른 SQL 쿼리를 생성하지 못했습니다.")
                st.code(query)
            else:
                try:
                    # 2) SQL 실행
                    conn = sqlite3.connect(DB_PATH)
                    df_result = pd.read_sql_query(query, conn)
                    conn.close()
                    
                    if not df_result.empty:
                        # 3) 시각화 코드 및 리포트 생성
                        viz_code = generate_viz_code(client, df_result, prompt)
                        report = generate_report(client, df_result, prompt)
                        
                        # 4) 결과 저장 (UI 유지를 위해 세션에 저장)
                        st.session_state.last_result = {
                            'query': query,
                            'explanation': explanation,
                            'df': df_result,
                            'viz_code': viz_code,
                            'report': report
                        }
                        
                        # 강제 리런하여 저장된 결과를 화면에 표시 (가장 깔끔한 방법)
                        st.rerun()
                        
                    else:
                        st.warning("조건에 맞는 데이터가 없습니다.")
                        st.session_state.messages.append({"role": "assistant", "content": "데이터 조회 결과가 없습니다."})
                        st.session_state.last_result = None
                        
                except Exception as e:
                    st.error(f"실행 중 오류 발생: {e}")
                    st.session_state.last_result = None