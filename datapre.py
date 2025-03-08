import io
import streamlit as st
import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama

st.title("Upload dataset")

uploaded_file = st.file_uploader("upload file plz", type=["csv"])

if uploaded_file is not None:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)

    df = st.session_state.df

    if st.button("info before cleaning"):

         infoData = {
            "column": df.columns,
            "Not null count": df.count().values,
            "Data type": [df[col].dtype for col in df.columns]
        }
         infoDataFrame = pd.DataFrame(infoData)
         st.subheader("Info of the Dataset 👇👇")
         st.dataframe(infoDataFrame)
    if st.button("Descripe"):
     st.subheader("Summay of data 👇👇")
     st.write(df.describe())


    if "num_rows" not in st.session_state:
        st.session_state.num_rows = 20 

  
    st.subheader("Data 👇👇")
    st.session_state.num_rows = st.number_input(
        "H.M rows Do u want to show ? 🙏",
        min_value=1,
        max_value=len(df),
        value=st.session_state.num_rows
    )
    if st.button("Show Data"):
        st.write(df.head(st.session_state.num_rows))

    
    numeric_cols = st.multiselect("PLZ choose the cols which contains numerical values only", options=df.columns.to_list(), help="choose numerical colsا")

    if numeric_cols:
        invalid_cols = {}
        invalid_data = {}
        for col in numeric_cols:
            invalid_values = df[col].apply(lambda x: isinstance(x, str) and not str(x).replace('.', '', 1).isdigit())
            if invalid_values.any():
                invalid_cols[col] = df[invalid_values].index.tolist()
                invalid_data[col] = df.loc[invalid_values, col].unique()

        if invalid_cols:
            st.subheader("Rows which contains invalid values")
            for col in invalid_cols:
                st.write(f"{col}")
                st.write(invalid_data[col].tolist())
        else:
            st.write("All of the cols are right ")

    if st.button("Replace invalid values with NAN"):
        if numeric_cols and invalid_cols:
            for col in invalid_cols:
                st.session_state.df[col] = pd.to_numeric(df[col], errors='coerce')
            st.success("Success replaced with NAN ")
        else:
            st.warning("لا توجد قيم غير صالحة للاستبدال.")
    if st.button("Info after cleaning 👇👇"):
        infoData = {
            "column": df.columns,
            "Not null count": df.count().values,
            "Data type": [df[col].dtype for col in df.columns]
        }
        infoDataFrame = pd.DataFrame(infoData)
        st.subheader("Info of the Dataset 👇👇")
        st.dataframe(infoDataFrame)

    st.subheader("Handling missing values")

    empty_cols = [col for col in df.columns if df[col].isnull().any()]

    if empty_cols:
        if "selectedcol" not in st.session_state:
            st.session_state.selectedcol = "اختر العمود"

        if "selected_action" not in st.session_state:
            st.session_state.selectedaction = "اختر الإجراء"

        selected_col = st.selectbox(
            "اختر العمود الذي يحتوي على قيم فارغة:",
            ["اختر العمود"] + empty_cols,
            index=0
        )

        if selected_col != "اختر العمود":
            st.session_state.selectedcol = selected_col

        col_type = df[st.session_state.selectedcol].dtype if st.session_state.selectedcol != "اختر العمود" else None

        if col_type is not None:
            if np.issubdtype(col_type, np.number):
                actions = ["اختر الإجراء", "Mean", "Median", "Mode"]
            else:
                actions = ["اختر الإجراء", "Mode"]

            selected_action = st.selectbox(
                "اختر الإجراء لتطبيقه على القيم الفارغة:",
                actions,
                index=0
            )

            if selected_action != "اختر الإجراء":
                st.session_state.selectedaction = selected_action

            if st.session_state.selectedcol != "اختر العمود" and st.session_state.selectedaction != "اختر الإجراء":
                st.write("**القيم قبل التغيير:**")
                st.write(df[st.session_state.selectedcol].head(10))

                if st.session_state.selectedaction == "Mean":
                    replacement_value = df[st.session_state.selectedcol].mean()
                elif st.session_state.selectedaction == "Median":
                    replacement_value = df[st.session_state.selectedcol].median()
                elif st.session_state.selectedaction == "Mode":
                    replacement_value = df[st.session_state.selectedcol].mode()[0]
                
                st.write(f"القيمة المُستخدمة للتعويض: {replacement_value}")

                if st.button("OK لتأكيد التغيير"):
                    df[st.session_state.selectedcol].fillna(replacement_value, inplace=True)
                    st.success(f"تم تطبيق التغيير على العمود {st.session_state.selectedcol}!")
                    st.write("**القيم بعد التغيير:**")
                    st.write(df[st.session_state.selectedcol].head(10))
    else:
        st.write("لا توجد قيم فارغة في البيانات.")

    st.subheader("Classification")

    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

    if not categorical_cols:
        st.warning("No categorical")
    else:
        st.session_state.selected_col = st.selectbox(
            "choose column",
            ["choose"] + categorical_cols,
            index=0
        )

        classification_option = st.radio("Choose a method for Classification:", ["Mapping", "One-Hot Encoding"])

        if classification_option == "Mapping":
            st.subheader("Choose mapping method:")
            if st.button("apply"):
             unique_values = df[st.session_state.selected_col].unique()
             mapping_dict = {value: index for index, value in enumerate(unique_values)}
             st.write(f"**Mapping applied:** {mapping_dict}")
             df[st.session_state.selected_col] = df[st.session_state.selected_col].map(mapping_dict)
             st.dataframe(df.head(st.session_state.num_rows))

        elif classification_option == "One-Hot Encoding":
            st.subheader("Applying One-Hot Encoding:")
            if st.button("apply"):
             one_hot_encoded_df = pd.get_dummies(df, columns=[st.session_state.selected_col])
             st.dataframe(one_hot_encoded_df.head(st.session_state.num_rows))
             df = one_hot_encoded_df 



    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
     label="Download Processed Data",
     data=csv,
     file_name='processed_data.csv',
     mime='text/csv',
   )