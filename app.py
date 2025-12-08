import streamlit as st
import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import matplotlib.font_manager as fm
import os
import platform
import re 

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
    elif system_name == 'Linux': # Streamlit Cloud (Linux)
        path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        
        if os.path.exists(path):
            fontprop = fm.FontProperties(fname=path, size=12)
            plt.rc('font', family=fontprop.get_name())
            print("✅ NanumGothic font set successfully.")
        else:
            print("⚠️ NanumGothic font not found. Please add 'fonts-nanum' to packages.txt")
            plt.rc('font', family='NanumGothic')
            
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
  - 도서관_부지면적 (FLOAT)
  - 도서관_건물_연면적 (FLOAT)
  - 도서관_서비스_제공면적 (FLOAT)
  - 총좌석수 (INTEGER)
  - 어린이_열람석 (INTEGER)
  - 노인및장애인_열람석 (INTEGER)

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
        'user': 'T4_이용자정보.csv',
        'service': 'T5_지식정보취약계층서비스.csv',
        'pop': 'T6_지역인구.csv'
    }
    
    # --------------------------------------------------------------------------------
# 수정된 initialize_database 함수 내부 로직
# --------------------------------------------------------------------------------

    missing_files = [path for path in csv_files.values() if not os.path.exists(path)]
    
    if missing_files:
        st.error(f"❌ 필수 파일이 누락되어 DB를 생성할 수 없습니다.\n누락된 파일: {missing_files}")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        progress_bar = st.progress(0)
        
        total = len(csv_files)
        
        for i, (table, path) in enumerate(csv_files.items()):
            df = read_csv_robust(path)
            
            if df.empty:
                st.warning(f"⚠️ {path} 파일은 존재하지만 데이터가 비어있습니다.")
                
            df.to_sql(table, conn, if_exists='replace', index=False)
            progress_bar.progress((i + 1) / total)

        conn.commit()
        conn.close()
        progress_bar.empty()
        st.toast("모든 데이터가 완벽하게 적재되었습니다!", icon="✅")
        return True
        
    except Exception as e:
        st.error(f"DB 생성 중 기술적 오류 발생: {e}")
        return False

# --------------------------------------------------------------------------------
# 4. LLM 및 분석 함수
# --------------------------------------------------------------------------------

def nl_to_sql(client, question):
    system_prompt = f"""
    당신은 SQLite 전문가입니다. 아래 스키마를 보고 질문을 SQL로 변환하세요.

    {FIXED_SCHEMA_INFO}

    [규칙]
    1. 결과 형식: 반드시 JSON 포맷 {{"sql": "SELECT ...", "explanation": "..."}} 만 출력하세요.
    2. 읽기 전용: SELECT 문만 사용 가능합니다. (INSERT, UPDATE, DELETE 금지)
    3. 테이블 조인(JOIN) 필수: 
       - `pop` 테이블과 다른 테이블들을 조인할 때는 반드시 복합키를 사용하세요.
       - 구문 예시: `ON pop.시도 = base_info.시도 AND pop.시군구 = base_info.시군구`
       - `base_info`, `service`, `holding`, `fac`, `user` 테이블끼리 join할 때는 `도서관코드`를 사용하세요.
    4. 비율 계산: 
       - 'A 대비 B' 또는 '비율'을 구할 때는 정수 나눗셈 오류를 방지하기 위해 `CAST`를 사용하세요.
       - 예: `CAST(SUM(B) AS FLOAT) / SUM(A)`
       - 비율 계산을 할 때 분모와 분자의 관계를 확실하게 이해하고 정확한 쿼리를 작성하세요. 
       - 예: 'B 대비 A의 비율'은 CAST(SUM(A) AS FLOAT) / SUM(B) 
    5. 그룹화(GROUP BY):
       - 지역별 통계를 구할 때는 `base_info.시도`, `base_info.시군구`로 그룹화하세요.
       - 집계 함수(SUM, AVG)를 적절히 사용하여 도서관별 데이터를 지역별로 합치세요.
    6. 제일 마지막에는 세미콜론(;)을 붙이세요.
    7. INSERT, UPDATE, DELETE 등 데이터 변경 구문은 절대 사용하지 마세요. (읽기 전용)
    8. 존재하지 않는 테이블이나 컬럼 이름을 지어내지 말고, 위에 정의된 스키마만 사용하세요. 스키마에 정의된 테이블명과 컬럼명을 글자 하나도 빼지 말고 그대로 사용하세요.
    9. SELECT나 WHERE 절에 사용된 컬럼이 있는 테이블은 반드시 FROM이나 JOIN 절에 포함되어야 합니다.
    
    [답변 예시]
    
    Q: "서울에 있는 도서관 이름 알려줘"
    A: {{
        "sql": "SELECT 도서관명 FROM base_info WHERE 시도 = '서울특별시';",
        "explanation": "서울특별시에 위치한 모든 도서관의 이름을 조회합니다."
    }}

    Q: "어린이 인구수 대비 어린이 서비스 이용수가 적은 지역(시군구) 3곳을 알려줘"
    A: {{
        "sql": "SELECT b.시도, b.시군구, (CAST(SUM(s.어린이서비스_이용수) AS FLOAT) / MAX(p.어린이인구)) AS 이용률 FROM base_info b JOIN pop p ON b.시도 = p.시도 AND b.시군구 = p.시군구 JOIN service s ON b.도서관코드 = s.도서관코드 GROUP BY b.시도, b.시군구 ORDER BY 이용률 ASC LIMIT 3;",
        "explanation": "지역별로 어린이 서비스 이용수 합계를 구한 뒤, 해당 지역의 어린이 인구수로 나누어 이용률이 가장 낮은 3곳을 추출합니다."
    }}
    
    Q: "장애인 관련 예산이 가장 많은 상위 5개 도서관과 그 지역을 알려줘"
    A: {{
        "sql": "SELECT b.시도, b.도서관명, s.취약계층관련예산_장애인 FROM base_info b JOIN service s ON b.도서관코드 = s.도서관코드 ORDER BY s.취약계층관련예산_장애인 DESC LIMIT 5;",
        "explanation": "서비스 테이블과 기본정보를 조인하여 장애인 예산이 가장 많은 순서대로 5개를 보여줍니다."
    }}
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
        
        if "sql" not in data:
            if "query" in data:
                data["sql"] = data["query"]
            elif "SQL" in data:
                data["sql"] = data["SQL"]
            else:
                data["sql"] = "-- SQL 생성 실패: AI가 올바른 형식을 반환하지 않음"
                if "explanation" not in data:
                    data["explanation"] = f"AI 응답 오류: {content}"
            
        if "explanation" not in data:
            data["explanation"] = "자동 생성된 쿼리입니다."
            
        return data

    except Exception as e:
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

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 
# --------------------------------------------------------------------------------


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Do+Hyeon&family=Gowun+Dodum&display=swap');

/* ✅ 헤더를 메인 컨텐츠 영역에만 고정 */
.header-container {
    position: fixed;
    top: 3.5rem;
    left: 0;
    right: 0;
    z-index: 9999;
    background: transparent;
    padding-left: 21rem;
    padding-right: 1rem;
    transition: padding-left 0.3s ease;
}

/* 사이드바가 있을 때 (기본 상태) */
section[data-testid="stSidebar"] ~ div .header-container {
    padding-left: 21rem;
}

/* 사이드바가 닫혔을 때 */
section[data-testid="stSidebar"][aria-expanded="false"] ~ div .header-container,
section[data-testid="stSidebar"].st-emotion-cache-1gwvy71 ~ div .header-container {
    padding-left: 1rem;
}

/* 모바일 대응 */
@media (max-width: 900px) {
    .header-container {
        padding-left: 1rem !important;
    }
}

/* ✅ 채팅 영역이 헤더에 안 가리게 밀기 */
.block-container {
    padding-top: 270px !important;
}

         
/* ✅ 디자인은 그대로 유지 */
.gradient-box {
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
    background: linear-gradient(45deg, #f0e8b5 0%, #62c0f9 100%); 
    color: white; 
    padding: 30px 40px; 
    border-radius: 15px; 
    font-family: 'Do Hyeon', sans-serif;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
}

.main-title {
    margin: 0; 
    font-size: 60px; 
    color: white;
    font-family: 'Do Hyeon', sans-serif;
    -webkit-text-stroke: 1px black;
}

.sub-title {
    font-family: 'Gowun Dodum', sans-serif; 
    margin: 5px 0 0 0; 
    font-size: 16px; 
    opacity: 0.9; 
    font-weight: normal;
    color: black;
}

.univ-info {
    text-align: right; 
    font-family: 'Gowun Dodum', sans-serif; 
    font-size: 15px; 
    opacity: 0.8; 
    font-weight: normal; 
    line-height: 1.5;
}

/* 모바일 대응 */
@media (max-width: 900px) {
    .header-container {
        left: 1rem;   /* ✅ 모바일에서는 사이드바 폭 제거 */
        right: 1rem;
    }
}
</style>

<div class="header-container">
    <div class="gradient-box">
        <div>
            <h1 class="main-title">💡 <span style="font-style: italic;">Light</span></h1>
            <p class="sub-title">도서관의 내일을 비추는 데이터 인사이트</p>
        </div>
        <div class="univ-info">
            <p style="margin: 0;">중앙대학교</p>
            <p style="margin: 0;">문헌정보학과</p>
        </div>
    </div>
</div>
            
""", unsafe_allow_html=True)



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
# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성
# --------------------------------------------------------------------------------

for msg in st.session_state.messages:
    # 1. 역할에 따라 아이콘(아바타) 결정
    if msg["role"] == "user":
        icon = "🙋‍♂️"  # 사용자
    else:
        icon = "💡"  # AI
        
    # 2. 메시지 표시
    with st.chat_message(msg["role"], avatar=icon):
        st.write(msg["content"])
        
        # 저장된 분석 결과(데이터, 그래프, 쿼리 등)가 있다면 탭으로 표시
        if "result" in msg:
            res = msg["result"]
            
            tab1, tab2, tab3, tab4 = st.tabs(["📋 데이터", "📈 시각화", "📝 리포트", "🔍 SQL"])
            
            with tab1:
                st.dataframe(res['df'])
                
            with tab2:
                if res['viz_code']:
                    try:
                        fig = plt.figure(figsize=(10, 6))
                        exec_globals = {'pd': pd, 'plt': plt, 'sns': sns, 'st': st}
                        exec_locals = {'df': res['df']}
                        exec("plt.show = lambda: None", exec_globals)
                        exec(res['viz_code'], exec_globals, exec_locals)
                        st.pyplot(plt.gcf())
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"시각화 복원 오류: {e}")
                        with st.expander("오류 코드 보기"):
                            st.code(res['viz_code'])
                            
            with tab3:
                st.markdown(res['report'])

            with tab4:
                st.info("이 결과를 만들기 위해 AI가 생성한 SQL입니다.")
                st.code(res['query'], language="sql")


# --------------------------------------------------------------------------------
# 6. 사용자 입력 처리
# --------------------------------------------------------------------------------

if prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지 화면 표시
    with st.chat_message("user", avatar="🙋‍♂️"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. AI 답변 처리
    with st.chat_message("assistant", avatar="💡"):
        message_placeholder = st.empty()
        
        with st.spinner("Light가 자료를 찾고 있습니다... 💡"):
            # 1) SQL 생성
            res_sql = nl_to_sql(client, prompt)
            query = res_sql['sql']
            explanation = res_sql['explanation']
            
            if "SELECT" not in query.upper():
                st.error("🚨 올바른 SQL 쿼리를 생성하지 못했습니다.")
                st.error(f"**에러 원인:** {explanation}") 
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
                        
                        # 4) 결과 데이터 포장
                        result_data = {
                            'query': query,
                            'explanation': explanation,
                            'df': df_result,
                            'viz_code': viz_code,
                            'report': report
                        }
                        
                        # 5) 화면에 즉시 보여주기
                        st.write(explanation)
                        
                        # [수정] 탭을 4개로 늘림 ("🔍 SQL" 추가)
                        tab1, tab2, tab3, tab4 = st.tabs(["📋 데이터", "📈 시각화", "📝 리포트", "🔍 SQL"])
                        
                        with tab1:
                            st.dataframe(df_result)
                            
                        with tab2:
                            try:
                                fig = plt.figure(figsize=(10, 6))
                                exec_globals = {'pd': pd, 'plt': plt, 'sns': sns, 'st': st}
                                exec_locals = {'df': df_result}
                                exec("plt.show = lambda: None", exec_globals)
                                exec(viz_code, exec_globals, exec_locals)
                                st.pyplot(plt.gcf())
                                plt.close(fig)
                            except:
                                st.error("시각화 실패")
                                
                        with tab3:
                            st.markdown(report)
                            
                        # [추가] 4번째 탭에 SQL 쿼리 표시
                        with tab4:
                            st.info("이 결과를 만들기 위해 AI가 생성한 SQL입니다.")
                            st.code(query, language="sql")
                        
                        # 6) 대화 기록에 저장
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": explanation, # 채팅창에는 설명만 텍스트로 저장
                            "result": result_data   # 복잡한 결과는 객체로 저장
                        })
                        
                    else:
                        st.warning("조건에 맞는 데이터가 없습니다.")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "데이터 조회 결과가 없습니다."
                        })
                        
                except Exception as e:
                    st.error(f"실행 중 오류 발생: {e}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"오류 발생: {e}"
                    })