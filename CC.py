# The target variable is `fraud_reported`, which can assume a value in {0,1}.<br/>
# The purpose of the model is to prioritize suspicious claims; we expect the target variable to be unbalanced
# with relatively few frauds.

import pandas as pd
from IPython.display import display, Markdown as md
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
import math
from scipy import stats

# Auxiliary functions taken from: https://www.kaggle.com/code/shakedzy/alone-in-the-woods-using-theil-s-u-for-survival/notebook
def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = stats.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def exe():
    INPUT_FILE = "claims.csv"
    df = pd.read_csv(INPUT_FILE)
    y_name = "fraud_reported"

    # ## 3.1 Feature analysis and engineering

    # ### 3.1.1 Missing values
    # What is the percentage of missing values in each column?
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(10)

    # We observe that only one column has 100% missing values and can be dropped.
    columns_to_drop = (missing_data[missing_data['Total'] > 10]).index
    df = df.drop(columns_to_drop, axis=1)


    # ### 3.1.2 Variable values
    # For each variable, we display the (max 10) most frequent values in [value, frequency] format.<br/>
    # It may be necessary to drop or modify columns.
    #
    # Number of samples in the dataset : 700 <br/>
    for c in df.columns:
        vc = df[c].value_counts(ascending=False).iloc[0:10]
        display(pd.DataFrame(vc))


    # We drop the following columns:
    # - `policy_number`: since it constitutes an ID
    # - `incident_location`: the incident may have happened at the same address in a different city or US state
    # - `insured_zip`: there is a different ZIP for each sample; we are already using different geographical information (state, city)
    # - `insured_hobbies`: it is a detail of dubious relevance at a very high level of granularity, we keep the rest of the information about the customer (education, occupation)
    # - `policy_bind_date`, `incident_date`: replaced by `insurance_duration`
    df_1 = df.drop(["policy_number", "incident_location", "insured_zip", "insured_hobbies"], axis=1)

    # We turn several variables from categorical to numerical, because there is information in the order :
    # - `policy_csl`: {100/300, 250/500, 500/1000}, it makes more sense as a numerical variable
    # - `incident_severity`: {Trivial/Minor/Major Damage, Total Loss},
    # - `police_report_available`: NO=0, ?=1, YES=2
    # - `property_damage`: NO=0, ?=1, YES=2
    df_1 = df_1.replace({"policy_csl": {"100/300": 1, "250/500": 2, "500/1000": 3},
                         "incident_severity": {"Trivial Damage": 0, "Minor Damage": 1, "Major Damage": 2, "Total Loss": 3},
                         "police_report_available": {"NO": 0, "?": 1, "YES": 2},
                         "property_damage": {"NO": 0, "?": 1, "YES": 2}})

    #  We create the following columns:
    #  -`insurance_duration`: it replaces the variables `policy_bind_date` and `incident_date` with their difference, in days
    a = pd.to_datetime(df["incident_date"], yearfirst=True)
    b = pd.to_datetime(df["policy_bind_date"], yearfirst=True)
    policy_time = a - b
    df_1["insurance_duration"] = policy_time.dt.days
    df_1 = df_1.drop(["incident_date", "policy_bind_date"], axis=1)
    df = df_1

    # ### 3.1.3 Numerical variables
    #
    # First, we compute the **correlation between the input variables and target variable**. We use the Spearman correlation instead of Pearson's, to avoid relying on the linearity of the relationship between variables.<br/>
    # If the correlation has a very low absolute value (here, |corr|<0.04), we eliminate the input variable

    # Find most important features relative to target
    numerical_vars = df.select_dtypes(include=np.number).columns.tolist()
    df_1 = df[numerical_vars]

    corr = df_1.corr(method='spearman')
    corr.sort_values([y_name], ascending=False, inplace=True)
    print(corr[y_name].to_string(name=False))

    relevant_vars =   [v for v in numerical_vars if abs(corr[y_name][v]) >= 0.04]
    relevant_vars.remove(y_name)
    vars_to_exclude = [v for v in numerical_vars if abs(corr[y_name][v]) < 0.04]
    df = df.drop(vars_to_exclude, axis=1)
    print("\n" + str(len(vars_to_exclude)) + " variables dropped: " + str(vars_to_exclude))
    print("\n" + str(len(relevant_vars)) + " variables kept: " + str(relevant_vars))

    # It is also opportune to examine the **collinearity between input variables**. In datasets, having two variables
    # with very high correlation (e.g. >0.8) means that they carry mostly the same information, in which case we have to keep only one.    #
    # Perfect collinearity is reached when one variable can be obtained as a linear combination of the other: <br/>
    # Given 2 input variables X1 and X2, $ \quad \forall i ( X1_i = \lambda_a X0_i + \lambda_b ) \quad$ and  $\quad correlation(X1_i, X2_i) = \pm 1$.
    #
    # Representing correlation values in a heatmap, we find that `injury_claim`, `property_claim`, and `vehicle_claim` are highly correlated to `total_claim_amount`. We decide to keep only `total_claim_amount`.

    corr = corr[relevant_vars].loc[relevant_vars]
    f, ax = plt.subplots(figsize=(10, 10))
    sns.set(font_scale=1.25)
    ax = sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12},
                     xticklabels=corr.index, yticklabels=corr.index)
    df = df.drop(["injury_claim", "property_claim", "vehicle_claim"], axis=1)


    # ### 3.1.4 Categorical variables
    #
    # We decide to examine two characteristics for each categorical variable:
    # - the number of unique values (i.e. categories)
    # - the strength of the association with the target variable, using Theil's U as a measure of nominal association
    #
    # Eventually we have to apply a one-hot encoder on categorical variables, causing an expansion of the number of features. It is best to first determine which variables are worth keeping.




    # In[10]:


    categorical_variables = []
    lts = []
    for c in df.columns:
        if df[c].dtype == pd.CategoricalDtype:  # we examine categorical variables here, not numerical
            categorical_variables.append(c)
            u = theil_u(df[c].tolist(), df[y_name].tolist())
            lts.append((c, df[c].value_counts().count(), round(u,5)))
    lts = sorted(lts, key=lambda tpl: tpl[2], reverse=True)
    c_info_df = pd.DataFrame(lts, columns=["Variable", "N. unique values", "Uncertainty_coeff"])
    display(c_info_df)

    # We decide to keep only the categorical variables with a high enough uncertainty coefficient (>0.01)
    # and with a number of unique values small enough (<20) to avoid an explosion of the number of features.
    cat_variables_to_drop = []
    for i,row in c_info_df.iterrows():
        v = row["Variable"]
        num_categories = row["N. unique values"]
        u_coeff = row["Uncertainty_coeff"]
        if (num_categories > 20) or (u_coeff < 0.01):
            cat_variables_to_drop.append(v)

    df = df.drop(cat_variables_to_drop, axis=1)

    df_1 = df.copy(deep=True)
    cat_variables_to_encode = list(set(categorical_variables).difference(set(cat_variables_to_drop)))
    label_enc = preprocessing.LabelEncoder()
    oh_enc = preprocessing.OneHotEncoder()
    num_samples = len(df.index)
    for cv in cat_variables_to_encode:
        df_1[cv] = label_enc.fit_transform(df[cv])
        # since weare encoding one feature at a time, we reshape the array
        df_1[cv] = oh_enc.fit_transform(np.array((df_1[cv])).reshape((num_samples,1))).todense()

    return df_1