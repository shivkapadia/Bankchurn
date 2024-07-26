import streamlit as st
import pandas as pd
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import re

# web page title
st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:")


# Load the preprocessor and model
@st.cache_resource
def load_preprocessor():
    return joblib.load('preprocessor.pkl')

@st.cache_resource
def load_model_from_file():
    return load_model('churn_model.h5')

preprocessor = load_preprocessor()
model = load_model_from_file()

# Function to load the CSV file
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the data
data = load_data('X_test.csv')

# Extract customer IDs for the dropdown list
customer_ids = data['CUSTOMER_ID'].unique()



# Streamlit interface
st.title("Customer Churn Prediction")

# Dropdown for selecting customer ID
selected_customer_id = st.selectbox("Select CUSTOMER_ID", customer_ids)

# Display customer details
if selected_customer_id:
    customer_details = data[data['CUSTOMER_ID'] == selected_customer_id]
    st.write("Customer Details:")
    st.write(customer_details)

# Prediction and LIME explanation
if st.button("Predict Churn"):
    # Convert customer details to DataFrame format
    customer_df = customer_details.drop(columns=['CUSTOMER_ID'])

    # Transform the data using the preprocessor
    transformed_data = preprocessor.transform(customer_df)

    # Predict churn probability
    churn_probability = model.predict(transformed_data)[0][0]

    # Format the probability as a percentage
    formatted_probability = f"{churn_probability * 100:.2f}%"

    # Display the probability with appropriate color
    if churn_probability < 0.5:
        st.success(f"Churn Probability: {formatted_probability}")
    else:
        st.error(f"Churn Probability: {formatted_probability}")

    # Get feature names after transformation
    transformed_feature_names = ['CR_PROD_CNT_IL', 'AMOUNT_RUB_CLO_PRC', 'APP_REGISTR_RGN_CODE', 'TURNOVER_DYNAMIC_IL_1M', 'CNT_TRAN_AUT_TENDENCY1M', 'SUM_TRAN_AUT_TENDENCY1M', 'AMOUNT_RUB_SUP_PRC', 'SUM_TRAN_AUT_TENDENCY3M', 'REST_DYNAMIC_FDEP_1M', 'CNT_TRAN_AUT_TENDENCY3M', 'REST_DYNAMIC_SAVE_3M', 'CR_PROD_CNT_VCU', 'REST_AVG_CUR', 'CNT_TRAN_MED_TENDENCY1M', 'AMOUNT_RUB_NAS_PRC', 'TRANS_COUNT_SUP_PRC', 'CNT_TRAN_CLO_TENDENCY1M', 'SUM_TRAN_MED_TENDENCY1M', 'TRANS_COUNT_NAS_PRC', 'CR_PROD_CNT_TOVR', 'CR_PROD_CNT_PIL', 'SUM_TRAN_CLO_TENDENCY1M', 'TURNOVER_CC', 'TRANS_COUNT_ATM_PRC', 'AMOUNT_RUB_ATM_PRC', 'TURNOVER_PAYM', 'AGE', 'CNT_TRAN_MED_TENDENCY3M', 'CR_PROD_CNT_CC', 'SUM_TRAN_MED_TENDENCY3M', 'REST_DYNAMIC_FDEP_3M', 'REST_DYNAMIC_IL_1M', 'SUM_TRAN_CLO_TENDENCY3M', 'LDEAL_TENOR_MAX', 'LDEAL_YQZ_CHRG', 'CR_PROD_CNT_CCFP', 'DEAL_YQZ_IR_MAX', 'LDEAL_YQZ_COM', 'DEAL_YQZ_IR_MIN', 'CNT_TRAN_CLO_TENDENCY3M', 'REST_DYNAMIC_CUR_1M', 'REST_AVG_PAYM', 'LDEAL_TENOR_MIN', 'LDEAL_AMT_MONTH', 'LDEAL_GRACE_DAYS_PCT_MED', 'REST_DYNAMIC_CUR_3M', 'CNT_TRAN_SUP_TENDENCY3M', 'TURNOVER_DYNAMIC_CUR_1M', 'REST_DYNAMIC_PAYM_3M', 'SUM_TRAN_SUP_TENDENCY3M', 'REST_DYNAMIC_IL_3M', 'CNT_TRAN_ATM_TENDENCY3M', 'CNT_TRAN_ATM_TENDENCY1M', 'TURNOVER_DYNAMIC_IL_3M', 'SUM_TRAN_ATM_TENDENCY3M', 'DEAL_GRACE_DAYS_ACC_S1X1', 'AVG_PCT_MONTH_TO_PCLOSE', 'DEAL_YWZ_IR_MIN', 'SUM_TRAN_SUP_TENDENCY1M', 'DEAL_YWZ_IR_MAX', 'SUM_TRAN_ATM_TENDENCY1M', 'REST_DYNAMIC_PAYM_1M', 'CNT_TRAN_SUP_TENDENCY1M', 'DEAL_GRACE_DAYS_ACC_AVG', 'TURNOVER_DYNAMIC_CUR_3M', 'MAX_PCLOSE_DATE', 'LDEAL_YQZ_PC', 'CLNT_SETUP_TENOR', 'DEAL_GRACE_DAYS_ACC_MAX', 'TURNOVER_DYNAMIC_PAYM_3M', 'LDEAL_DELINQ_PER_MAXYQZ', 'TURNOVER_DYNAMIC_PAYM_1M', 'CLNT_SALARY_VALUE', 'TRANS_AMOUNT_TENDENCY3M', 'MED_DEBT_PRC_YQZ', 'TRANS_CNT_TENDENCY3M', 'LDEAL_USED_AMT_AVG_YQZ', 'REST_DYNAMIC_CC_1M', 'LDEAL_USED_AMT_AVG_YWZ', 'TURNOVER_DYNAMIC_CC_1M', 'AVG_PCT_DEBT_TO_DEAL_AMT', 'LDEAL_ACT_DAYS_ACC_PCT_AVG', 'REST_DYNAMIC_CC_3M', 'MED_DEBT_PRC_YWZ', 'LDEAL_ACT_DAYS_PCT_TR3', 'LDEAL_ACT_DAYS_PCT_AAVG', 'LDEAL_DELINQ_PER_MAXYWZ', 'TURNOVER_DYNAMIC_CC_3M', 'LDEAL_ACT_DAYS_PCT_TR', 'LDEAL_ACT_DAYS_PCT_TR4', 'LDEAL_ACT_DAYS_PCT_CURR', 'CLNT_TRUST_RELATION_BROTHER', 'CLNT_TRUST_RELATION_DAUGHTER', 'CLNT_TRUST_RELATION_FATHER', 'CLNT_TRUST_RELATION_FRIEND', 'CLNT_TRUST_RELATION_MOTHER', 'CLNT_TRUST_RELATION_OTHER', 'CLNT_TRUST_RELATION_RELATIVE', 'CLNT_TRUST_RELATION_SISTER', 'CLNT_TRUST_RELATION_SON', 'CLNT_TRUST_RELATION_Близкий ро', 'CLNT_TRUST_RELATION_Брат', 'CLNT_TRUST_RELATION_Дальний ро', 'CLNT_TRUST_RELATION_Дочь', 'CLNT_TRUST_RELATION_Друг', 'CLNT_TRUST_RELATION_Жена', 'CLNT_TRUST_RELATION_Мать', 'CLNT_TRUST_RELATION_Муж', 'CLNT_TRUST_RELATION_Отец', 'CLNT_TRUST_RELATION_Сестра', 'CLNT_TRUST_RELATION_Сын', 'CLNT_TRUST_RELATION_мать', 'APP_MARITAL_STATUS_ ', 'APP_MARITAL_STATUS_C', 'APP_MARITAL_STATUS_D', 'APP_MARITAL_STATUS_M', 'APP_MARITAL_STATUS_N', 'APP_MARITAL_STATUS_T', 'APP_MARITAL_STATUS_V', 'APP_MARITAL_STATUS_W', 'APP_MARITAL_STATUS_d', 'APP_MARITAL_STATUS_m', 'APP_MARITAL_STATUS_t', 'APP_MARITAL_STATUS_v', 'APP_MARITAL_STATUS_w', 'APP_KIND_OF_PROP_HABITATION_JO', 'APP_KIND_OF_PROP_HABITATION_NPRIVAT', 'APP_KIND_OF_PROP_HABITATION_OTHER', 'APP_KIND_OF_PROP_HABITATION_RENT', 'APP_KIND_OF_PROP_HABITATION_SO', 'CLNT_JOB_POSITION_TYPE_MANAGER', 'CLNT_JOB_POSITION_TYPE_SELF_EMPL', 'CLNT_JOB_POSITION_TYPE_SPECIALIST', 'CLNT_JOB_POSITION_TYPE_TOP_MANAGER', 'APP_DRIVING_LICENSE_N', 'APP_DRIVING_LICENSE_Y', 'APP_EDUCATION_A', 'APP_EDUCATION_AC', 'APP_EDUCATION_AV', 'APP_EDUCATION_E', 'APP_EDUCATION_H', 'APP_EDUCATION_HH', 'APP_EDUCATION_HI', 'APP_EDUCATION_I', 'APP_EDUCATION_S', 'APP_EDUCATION_SS', 'APP_EDUCATION_UH', 'APP_EDUCATION_US', 'APP_EDUCATION_a', 'APP_EDUCATION_e', 'APP_EDUCATION_h', 'APP_EDUCATION_i', 'APP_EDUCATION_s', 'APP_TRAVEL_PASS_N', 'APP_TRAVEL_PASS_Y', 'APP_CAR_N', 'APP_CAR_Y', 'APP_POSITION_TYPE_MANAGER', 'APP_POSITION_TYPE_SELF_EMPL', 'APP_POSITION_TYPE_SPECIALIST', 'APP_POSITION_TYPE_TOP_MANAGER', 'APP_EMP_TYPE_INTER', 'APP_EMP_TYPE_IP', 'APP_EMP_TYPE_PRIVATE', 'APP_EMP_TYPE_STATE', 'APP_COMP_TYPE_INTER', 'APP_COMP_TYPE_IP', 'APP_COMP_TYPE_PRIVATE', 'APP_COMP_TYPE_STATE', 'PACK_101', 'PACK_102', 'PACK_103', 'PACK_104', 'PACK_105', 'PACK_107', 'PACK_108', 'PACK_109', 'PACK_301', 'PACK_K01', 'PACK_M01', 'PACK_O01']

    # Check the number of features in the transformed data
    assert transformed_data.shape[1] == len(transformed_feature_names), f"Number of features does not match, features: {transformed_data.shape[1]}, expected: {len(transformed_feature_names)}"

    # Create a predict function that returns class probabilities
    def predict_proba(data):
        preds = model.predict(data)
        # Convert single class probability to two class format expected by LIME
        return np.hstack((1 - preds, preds))

    explainer = LimeTabularExplainer(transformed_data,
        feature_names=transformed_feature_names,
        class_names=['Not Churn', 'Churn'],
        mode='regression')


    # Now explain a prediction
    exp = explainer.explain_instance(transformed_data[0], model.predict,
            num_features=10)
    
    # Get weights for the explanation
    weights = exp.as_list()

    # Define a regex pattern to match the number
    pattern = r"[-+]?\d*\.\d+|\d+"

    # iterate by each key and extract all the values to a list
    feature_names_exp = [key[0] for key in exp.as_list()]
    feature_names = [key.split(' <= ')[0] for key in feature_names_exp]
    weights = [key.split(' <= ')[1] for key in feature_names_exp]
    feature_weights = []
    for x in weights:
        match = re.search(pattern, x)
        if match:
            feature_weights.append(float(match.group()))
        else:
            feature_weights.append(None)
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Colors: green for positive, red for negative
    colors = ['green' if weight > 0 else 'red' for weight in feature_weights]

    # Plot bars
    ax.barh(feature_names, feature_weights, color=colors)

    # Add labels and title
    ax.set_xlabel('Feature Weight')
    ax.set_title('Feature Importance Explanation')
    
    # Display the plot
    plt.tight_layout()  # Ensure the layout is tight so the chart fits well
    st.pyplot(fig)