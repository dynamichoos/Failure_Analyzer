#from turtle import window_width
from sklearn.datasets import *
from sklearn import preprocessing, tree
from dtreeviz.trees import dtreeviz
import base64
import graphviz as graphviz

import pandas as pd
import plotly.express as px
import streamlit as st
import os

import streamlit.components.v1 as components


import sys
import subprocess

def pip_install(package):
    subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", package])
    
pip_install('python-pydot python-pydot-ng graphviz')

#from wcwidth import wcswidth
#import conversion_excel


### source: https://github.com/Sven-Bo/streamlit-sales-dashboard ###
### emojis : https:/www.webfx.com/tools/emoji-cheat-sheet/

## UN data : http://data.un.org/Data.aspx?d=FAO&f=itemCode%3a800

## 관련 함수

def df_label_count_limit(df, target_col):
    
    ### Top 10에 대한 불량만 조사 ### (10개 이상 시에 dtreeviz 사용 불가 <- 추가 내용 있는지 확인 필요)
    df_target_counts = df[target_col].value_counts().sort_values(ascending=False)
    limit = 10
    screened_failures =[]

    if len(df_target_counts) > limit:
        screened_failures = list(df_target_counts.index)[limit:]
        print(f'all screened_failures :{screened_failures}')
        
    for screened_failure in screened_failures:
        df = df[df[target_col] != screened_failure]
        print(f'{screened_failure} is screened')
        
    return df

def get_continous_df(df, continuous_cols):
    
    ### 연속형 데이터에 대해서 int 와 10진수로 변경 ####
    for i in continuous_cols:
        print(i)
        df[i] = df[i].fillna(0)
    
        if i == '주행거리':
            df[i] = df[i].astype(int)

        elif i != '주행거리':
            df[i] = df[i].apply(lambda x:int(str(x),16))
            
    return df

def get_dummies_df(df, label_col):
    
    if label_col == '소비전류 (A)\nBAT=14V':
        
        df[label_col] = df[label_col].fillna('nan')
        result = df[label_col].astype(str).str.upper().str.replace('고정','고정/').str.replace('/',',').str.replace('\n',',').str.split(',')
        result = result.apply(lambda x:pd.Series(x))
        result = result.stack()
        result = result.astype(str).str.strip()
        result = get_dummies_df_current(df, label_col)
        result = result.reset_index(level=1,drop=True).to_frame(label_col)
        #result = df.merge(result, left_index=True, right_index=True, how='left')
    
        result_ohe = pd.get_dummies(result, columns=[label_col])
        result = result_ohe.groupby(level=0).sum()
        df = df.drop(columns=[label_col])
        df = df.join(result)

    else:
        #result = df[label_col].str.split(',')
        df[label_col] = df[label_col].fillna('nan')
        result = df[label_col].astype(str).str.upper().str.replace('/',',').str.replace('\n',',').str.split(',')
        result = result.apply(lambda x:pd.Series(x))
        result = result.stack()
        result = result.astype(str).str.strip()
        result = result.reset_index(level=1,drop=True).to_frame(label_col)
        #result = df.merge(result, left_index=True, right_index=True, how='left')
    
        result_ohe = pd.get_dummies(result, columns=[label_col])
        result = result_ohe.groupby(level=0).sum()
        df = df.drop(columns=[label_col])
        df = df.join(result)
    
    return df

#def file_selector(folder_path='./raw_excel_file/'):
#./resource/21_22_고품접수 TF_정리_LIST_FCM_KLK_220330_고객(HKMC공유 예정)_내용 수정_방향.xlsx
#def file_selector(folder_path='D:/python_project/5.FA_AI/raw_excel_file/'):
def file_selector(folder_path='./resource/'):
    
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('- Select an excel file', filenames)
    return os.path.join(folder_path, selected_filename)


@st.cache
def get_raw_data_from_excel(file_name, sheet_name):
    
    df = pd.read_excel(file_name, sheet_name=sheet_name, header=1)
    df.dropna(subset=['고품No.'],inplace=True)
    
    return df

## 1. 환경 셋업 및 xlsx 데이터 로드

#### ------------------ Set page ------------------- ###

title_ = 'Failure Analyzer'

st.set_page_config(page_title = title_ ,
                   page_icon=':bar_chart:',
                   layout = 'wide'
)

#### ------------------ SIDEBAR ------------------- ###

st.sidebar.header('1. Open file')

file_name = file_selector()
#file_name = './resource/21_22_고품접수 TF_정리_LIST_FCM_KLK_220330_고객(HKMC공유 예정)_내용 수정_방향.xlsx'
#file_name = "https://github.com/dynamichoos/Failure_Analyzer/blob/main/resource/FA_LIST.xlsx"
st.write('- You selected the file: `%s`' % file_name)

## Set sheet_name ##
sheet_name = st.sidebar.selectbox(
    '- Select sheet name', 
    ['FCM_KLK','tbd']
)

df_raw = get_raw_data_from_excel(file_name, sheet_name)
df = df_raw.copy()


column_lists = df.columns.to_list()
column_lists.insert(0,'<select>')

## 2. target(label) column 설정
##   a. 보고싶은 target label만 필터링
##   b. top 10기준 외 항목만 선택

st.sidebar.header('2. Set Label')
label_col = st.sidebar.selectbox(
    '- Select Label column', 
    column_lists, index = column_lists.index('Defect2')
)

label_col_filter = st.sidebar.multiselect(
    '- Select targeting failures(maximum 10ea): ',
    options = sorted(df[label_col].unique()),
    default = sorted(df[label_col].unique()),
)

query_msg = r'{label_col} == @label_col_filter'
df =  df.query(
    #query_msg
    "Defect2 == @label_col_filter"
)

## 대상 label을 10개 항목으로 제한
df = df_label_count_limit(df,label_col)

## 3. feature column 설정 (filter column으로 구성)
##   a. 연속형
##   b. 이산형
##   c. 그외 parsing 필요형

####### 1. Filtering이 필요한 column #######
st.sidebar.header('3. Set Features')
filter_cont = st.sidebar.multiselect(
    "- Select columns(continous features): ",
    options = sorted(df.columns.to_list()),
    default = ['주행거리','Booting','Pending','Running'],
    #default = None
)

print(filter_cont)

filter_disc = st.sidebar.multiselect(
    "- Select columns(discrete features): ",
    options = sorted(df.columns.to_list()),
    default = ['ITEM','차종','Boot','App','EYEQ','MCU','DDR','Flash'],
    #default = None
)

print(filter_disc)

filter_raw = st.sidebar.multiselect(
    "- Select columns(raw features): ",
    options = sorted(df.columns.to_list()),
    default = ['Simulator\nDTC','Internal Error', '분석 결과','SA TEST','내부 5회','양산','Boundary TEST'],
    #default = None
)

### 선택된 features 를 기준으로 filtering ##

features_total = []
features_total.extend(filter_cont)
features_total.extend(filter_disc)
features_total.extend(filter_raw)
#features_total.remove(target_col)
#features_total.extend(label_col)

df_label = df[label_col]
df = df[features_total]
df[label_col] = df_label

print(features_total)
print(df)

df_selected = df.copy()


############################################################
### 1. radio button으로 feature 내용 기반 필터링 여부 추가 ###
### 2. 필더링 진행 시 아래의 내용과 같이 항목 추가          ###

st.sidebar.header('4. Set Features w/ advanced filter')

option = st.sidebar.selectbox('Choose the type of filter!',
                      ('Normal Filter', 'Advanced Filter'))
st.write('- You selected the filter:', option)

if option == 'Advanced Filter':
    
    ####### 1. Filtering이 필요한 column #######
    filter_1 = st.sidebar.multiselect(
        "Select filter column(usually Item): ",
        options = sorted(df.columns.to_list()),
        #default = 'MISO_Addr'
        default = 'ITEM'
    )

    print(filter_1)

    ####### 2. Filtering이 된 column에서 Filtering 하려는 값 선택 (1번에서 선택한 column만큼 필터 갯수가 추가됨) #######
    for idx, filter_ in enumerate(filter_1):
        globals()['filter_No_{}'.format(idx)] = st.sidebar.multiselect(
            f"{idx+1}. Select data range of {filter_}:", 
            options = sorted(df[filter_].unique()),
            #default = ('주행거리','Booting','Pending','Running')
            default = sorted(df[filter_].unique()),
        )

    query_ = ''

    for idx, filter_ in enumerate(filter_1):
        query_ += f"{filter_} == @filter_No_{idx}"

        if idx < len(filter_1) - 1:
            query_ += ' & '

    print(query_)
    df_selected = df.query(query_)
    print(df_selected)

############################################################

## 4. feature selection 2차
##   a. 연속형 - 10진수로 변환
##   b. 이산형 - one_hot_encoding
##   c. 그외형 - parsing & one_hot_encoding
##   d. 개별 유형 - 개별 coding 수행 (eg> 전류)

### 연속형 데이터(continous features) 기본 전처리 ###
df_selected = get_continous_df(df_selected, filter_cont)

### 이산형 데이터(discrete features) 기본 전처리 ###
for features_disc_ in filter_disc:
    df_selected[features_disc_] = df_selected[features_disc_].astype('str')

# one_hot_encoding_Dummy 추가 ####
df_selected = pd.get_dummies(df_selected, columns=filter_disc, drop_first=False)

### raw features 전처리 ###
for dummy_target_col in filter_raw:
    print(dummy_target_col)
    df_selected = get_dummies_df(df_selected,dummy_target_col)

features_total = list(df_selected.columns)
features_total.remove(label_col)

## 5. Classifier 적용
##   a. fit model


#@st.cache
def visualization_dtreeviz(df_selected, label_col):
    
    label_col_encoder = preprocessing.LabelEncoder()
    label_col_encoder.fit(df_selected[label_col])
    df_selected['target'] = label_col_encoder.transform(df_selected[label_col])

    #classifier = tree.DecisionTreeClassifier(random_state=1234, max_depth=5)
    classifier = tree.DecisionTreeClassifier(random_state=1234)

    #classifier.fit(df.iloc[:,:4], df[target_col])
    classifier.fit(df_selected[features_total], df_selected['target'])

    viz = dtreeviz(classifier,
            #df.iloc[:,:4],
            df_selected[features_total],
            df_selected['target'],
            target_name='Failures',
            #feature_names=df.columns[0:4],
            feature_names=features_total,
            class_names=list(label_col_encoder.classes_),
            fontname = 'gulim',
            #orientation='LR',
            #fancy=False,
            )

    return viz


def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    # Encode as base 64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True)

## 6. Visiualization (by dtreeviz)
##   ** streamlit에 사용을 위한 추가 조사 필요

## ---------- Mainpage --------

st.title(f":bar_chart: {title_}")

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
#selected_item = st.radio("Plot by features", sorted(df.columns.to_list()),index=8)
selected_item = st.radio("Plot by features", sorted(['Defect2','ITEM','차종','Simulator\nDTC']),index=1,)

try:

    st.text("Descriptive statistics by feature")
    st.dataframe(df.groupby(by=[selected_item])[label_col].describe().reset_index())

    fig_data3 = px.bar(
        df, 
        x = selected_item,
        #y = label_col,
        title = f"<b>[Bar plot] x-{selected_item} // y- # of cnt // Hue-{label_col}</b>",
        #color_discrete_sequence=["#0083B8"] * len(df_selection),
        color =label_col,
        #hover_data=df.colmuns,
        hover_data=df[['Defect2','ITEM','차종','Simulator\nDTC']],
        template = "plotly_white",
    )
    
    fig_data3.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )

    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig_data3, use_container_width=True)


    fig_data4 = px.bar(
        df, 
        x = label_col,
        #y = label_col,
        title = f"<b>[Bar plot] x-{label_col} // y- # of cnt // Hue-{selected_item}</b>",
        #color_discrete_sequence=["#0083B8"] * len(df_selection),
        color =selected_item,
        #hover_data=df.colmuns,
        hover_data=df[['Defect2','ITEM','차종','Simulator\nDTC']],
        template = "plotly_white",
    )
    
    fig_data4.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )
    
    right_column.plotly_chart(fig_data4, use_container_width=True)


except:
    print('filter structure is not correct for multi indexes')



st.markdown('##')
st.markdown('---')


st.header(":palm_tree: Failure Classifier by analyzed results")
st.markdown('####')

print(df_selected)

###################################################################

import os
#common_path = os.path.commonpath([path for path in os.environ["PATH"].split(';') if 'Anaconda' in path])
#check_all_path = os.path.commonpath([path for path in os.environ["PATH"].split(';')] if 'graphviz' in path])
#check_all_path = os.path.commonpath([path for path in os.environ["PATH"].split(';')])


print('Here!! base path!!')
print(os.environ["PATH"])
'''
exec("where dot.exe")

common_path = "/home/appuser/venv/bin"
dot_path = os.path.join(common_path, 'Library', 'bin', 'graphviz')
os.environ["PATH"] += os.pathsep + dot_path


common_path = "/home/appuser/venv/lib/python3.7/site-packages/"
dot_path = os.path.join(common_path, 'graphviz')
os.environ["PATH"] += os.pathsep + dot_path

'''
common_path = "/home/appuser/venv/lib/python3.7/site-packages/graphviz/"
os.environ["PATH"] += os.pathsep + common_path

###################################################################

viz=visualization_dtreeviz(df_selected,label_col)
svg=viz.svg()


mid_column = st.columns(1)
svg_write(svg)

#mid_column = st.columns(1)

## to save result
viz.save("decision_tree_in_streamlit.svg")



## 7. 저장
##   a. SVG 저장
##   b. preprocess된 data 함께 저장

left_down_button, _, _, _, right_down_button = st.columns(5)

with left_down_button:
    st.download_button(
    label = "Download Filtered data",
    data = df_selected.to_csv(),
    file_name = 'fitlered_df.csv',
    mime = 'text/csv',
    )

# with right_down_button:
#     st.download_button(
#     label = "Download fault classification",
#     data = df_msg_selection.to_csv(),
#     file_name = 'SPI_signal_mapped(selected)_df.csv',
#     mime = 'text/csv',
#     )

with right_down_button:
    result = st.button("Download fault classification")
    if result:
        viz.save('decision_tree_in_streamlit.svg')
