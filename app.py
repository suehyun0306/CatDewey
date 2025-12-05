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
  - 도서관 부지 면적 (FLOAT)
  - 도서관 건물 연면적 (FLOAT)
  - 도서관 서비스 제공 면적 (FLOAT)
  - 총 좌석수 (INTEGER)
  - 어린이 열람석 (INTEGER)
  - 노인 및 장애인 열람석 (INTEGER)

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

    # 1. [수정] 누락된 파일이 있는지 먼저 검사합니다.
    missing_files = [path for path in csv_files.values() if not os.path.exists(path)]
    
    # 2. [수정] 하나라도 없으면 에러를 띄우고 즉시 중단합니다. (거짓말쟁이 방지)
    if missing_files:
        st.error(f"❌ 필수 파일이 누락되어 DB를 생성할 수 없습니다.\n누락된 파일: {missing_files}")
        # 파일이 없으면 기존 DB라도 쓰게 할지, 아예 멈출지 결정해야 하는데
        # '업로드가 잘못된 것을 알아야 한다'는 선생님 의견에 따라 여기서 멈춥니다.
        return False

    # 3. 모든 파일이 존재할 때만 아래 로직이 실행됩니다.
    try:
        conn = sqlite3.connect(DB_PATH)
        progress_bar = st.progress(0)
        
        # 이제 existing_files 대신 원래 csv_files를 그대로 씁니다. (다 있는 걸 확인했으니까요)
        total = len(csv_files)
        
        for i, (table, path) in enumerate(csv_files.items()):
            df = read_csv_robust(path)
            
            # 데이터프레임이 비어있는 경우도 체크하면 더 좋습니다 (선택사항)
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

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 고정형)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 강력 고정형 fixed)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 사이드바 안 가리는 버전)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 반응형 Sticky 버전)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 사이드바 반응형 + 상단 여백 제거)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 기본 헤더 숨김 + 완벽한 Sticky)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 기본바 제거 + Sticky 고정)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 디자인만 적용된 기본 버전)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (상단 헤더 - 디자인만 적용된 기본 버전)
# --------------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Do+Hyeon&family=Gowun+Dodum&display=swap');

/* ✅ 헤더를 메인 컨텐츠 영역에만 고정 */
.header-container {
    position: fixed;
    top: 3.5rem;               /* Streamlit 상단바 아래 */
    left: 18rem;               /* ✅ 사이드바 너비만큼 밀기 */
    right: 1rem;
    z-index: 9999;
    background: transparent;
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
    background: linear-gradient(45deg, #337de6 0%, #149c9f 100%); 
    color: white; 
    padding: 30px 40px; 
    border-radius: 15px; 
    font-family: 'Do Hyeon', sans-serif;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
}

.main-title {
    margin: 0; 
    font-size: 32px; 
    color: white;
    font-family: 'Do Hyeon', sans-serif;
}

.sub-title {
    font-family: 'Gowun Dodum', sans-serif; 
    margin: 5px 0 0 0; 
    font-size: 16px; 
    opacity: 0.9; 
    font-weight: normal;
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
            <h1 class="main-title">📮 <span style="font-style: italic;">사서함 : 사서와 함께</span></h1>
            <p class="sub-title">지적자유 전문상담 챗봇</p>
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
# 5. Streamlit 화면 구성 (중간 부분 수정)
# --------------------------------------------------------------------------------

# 이전 대화 및 분석 결과 출력
# --------------------------------------------------------------------------------
# 5. Streamlit 화면 구성 (중간 부분 - 대화 기록 출력)
# --------------------------------------------------------------------------------

for msg in st.session_state.messages:
    # 1. 역할에 따라 아이콘(아바타) 결정
    if msg["role"] == "user":
        icon = "🙋‍♂️"  # 사용자: 손 든 사람
    else:
        icon = "🦉"  # AI: 부엉이 사서
        
    # 2. 결정된 아이콘을 넣어 메시지 표시 (기존 코드 대신 이 부분을 씁니다)
    with st.chat_message(msg["role"], avatar=icon):
        st.write(msg["content"])
        
        # 만약 이 메시지에 분석 결과(데이터, 그래프 등)가 저장되어 있다면 그려줍니다.
        if "result" in msg:
            res = msg["result"]
            
            # 탭 생성
            tab1, tab2, tab3 = st.tabs(["📋 데이터", "📈 시각화", "📝 리포트"])
            
            with tab1:
                st.dataframe(res['df'])
                
            with tab2:
                # 저장된 코드로 그래프 그리기
                if res['viz_code']:
                    try:
                        # 1. 그림 그릴 도화지(Figure)를 새로 꺼냅니다.
                        fig = plt.figure(figsize=(10, 6))
                        
                        # 2. 실행 환경 설정
                        exec_globals = {'pd': pd, 'plt': plt, 'sns': sns, 'st': st}
                        exec_locals = {'df': res['df']}
                        
                        # 3. plt.show() 무력화 (에러 방지용)
                        exec("plt.show = lambda: None", exec_globals)
                        
                        # 4. 시각화 코드 실행
                        exec(res['viz_code'], exec_globals, exec_locals)
                        
                        # 5. 그려진 그림을 화면에 출력
                        st.pyplot(plt.gcf())
                        
                        # 6. 메모리 정리를 위해 도화지 닫기
                        plt.close(fig)
                        
                    except Exception as e:
                        st.error(f"시각화 복원 오류: {e}")
                        # 혹시 모르니 코드도 보여줌
                        with st.expander("오류 코드 보기"):
                            st.code(res['viz_code'])
                            
            with tab3:
                st.info(res['report'])
                with st.expander("🔍 사용된 SQL 쿼리 확인"):
                    st.code(res['query'], language="sql")
# --------------------------------------------------------------------------------
# 6. 사용자 입력 처리 (마지막 부분 수정)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 6. 사용자 입력 처리 (마지막 부분 - 실시간 대화)
# --------------------------------------------------------------------------------

if prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지 화면 표시 (🙋‍♂️ 아이콘 추가)
    with st.chat_message("user", avatar="🙋‍♂️"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. AI 답변 처리 (🦉 아이콘 추가)
    with st.chat_message("assistant", avatar="🦉"):
        message_placeholder = st.empty()
        
        with st.spinner("부엉이 사서가 자료를 찾고 있습니다... 🦉"): # 멘트도 귀엽게 변경!
            # 1) SQL 생성
            res_sql = nl_to_sql(client, prompt)
            query = res_sql['sql']
            explanation = res_sql['explanation']
            
            if "SELECT" not in query.upper():
                st.error("올바른 SQL 쿼리를 생성하지 못했습니다.")
                st.code(query)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "SQL 생성 실패: " + query
                })
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
                            'df': df_result,
                            'viz_code': viz_code,
                            'report': report
                        }
                        
                        # 5) 화면에 즉시 보여주기
                        st.write(explanation)
                        tab1, tab2, tab3 = st.tabs(["📋 데이터", "📈 시각화", "📝 리포트"])
                        
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
                            st.info(report)
                        
                        # 6) 대화 기록에 저장
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": explanation,
                            "result": result_data 
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