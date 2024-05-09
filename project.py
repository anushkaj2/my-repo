import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import shap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

def get_gene_data():
    # run a for loop across all files and concatenate values to a dataframe
    df = pd.read_csv(r"/home/ibab/autism_ged/GSE42133_family.xml/GSM1033105-tbl-1.txt", sep="\t", header=None, names=["gene", "gsm1033105"])
    for i in range(106,256):
        if i!=151 and i!=210 and i!=226 and i!=252:
            df_new = pd.read_csv(rf"/home/ibab/autism_ged/GSE42133_family.xml/GSM1033{i}-tbl-1.txt", sep="\t", header=None, names=["gene", f"gsm1033{i}"])
            df = pd.concat([df, df_new[f"gsm1033{i}"]], axis=1)
    return df

def get_samples_data():
    # get data from files that have already been filtered through linux commands
    df = pd.read_csv(r"/home/ibab/autism_ged/gse42133/newfile1.txt", sep="\t", header=None, names=["Sample"])
    df_new = pd.read_csv(rf"/home/ibab/autism_ged/gse42133/newfile2.txt", sep="\t", header=None, names=["y"])
    df = pd.concat([df, df_new], axis=1)
    return df

def probe2gene_mapping(df):
    mapping_df = pd.read_csv(r"/home/ibab/autism_ged/probe2gene.csv")   # there are 13,360 nan values which will be removed from the dataframe
    mapping_list = list(mapping_df.iloc[:, 1])
    mapping_dict = {index : value for index, value in enumerate(mapping_list)}
    df.rename(columns=mapping_dict, inplace=True)
    df = df.loc[:, df.columns != "no_value"]       # remaining cols: 47323-13360 = 33963
    number_unique_columns = len(np.unique(df.columns))      # there are 20909 unique columns in the data frame which are unique genes
    df = df.T       # transpose to make group by easier to calculate mean
    df = df.groupby(level=0).mean().transpose()     # after calculating means transpose again
    return df

def save_df():
    # combine both samples and gene expression data
    df = get_gene_data()  # get gene expression level data
    labels = df.iloc[:, 0]
    df = df.T  # transpose
    df = df.iloc[1:, :]  # remove the label row from df
    df = probe2gene_mapping(df)
    sample = get_samples_data()  # get data on the samples
    df = df.reset_index(drop=True)  # as the concatenation is not happening, we drop index of the df
    df = pd.concat([df, sample["y"]], axis=1)  # concatenate the gene expression levels with the sample labels
    df.to_csv(r"/home/ibab/autism_ged/autism_gene_expression.csv", index=False)
    labels.to_csv(r"/home/ibab/autism_ged/labels.csv", index=False)
    return "Data Saved"

def load_data():
    df = pd.read_csv(r"/home/ibab/autism_ged/autism_gene_expression.csv")
    x = df.iloc[:, :-1]  # Gene expression levels corresponding to 47,323 probes for 147 sample
    y = df.iloc[:, -1]  # Class labels: ASD or Control
    return df, x, y

def eda(df, x, y):
    print(df.describe().to_string())        # Descriptive statistics (not very useful here as data as big)
    print(y.value_counts())     # Checks for class imbalances: ASD 91, Control 56
    print(x.shape)      # OUTPUT: (147, 20909)
    print(y.shape)      # OUTPUT: (147,)
    if df.isnull() is True:     # Checks for the presence of null values
        print("Null values are present somewhere in the dataframe")
    else:
        print("Null values are not present")

def data_standardization(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return pd.DataFrame(x)


def k_fold_validation(model, x, y):
    k = 10
    x = x.astype(str)
    cv_scores = []

    kf = KFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        x_train = x_train.astype(float)
        x_test = x_test.astype(float)

        # fit model
        model.fit(x_train, y_train)

        cv_score = model.score(x_test, y_test)
        cv_scores.append(cv_score)

    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    return mean_acc, std_acc

def run_models(model, x, y):
    mean_acc, std_acc = k_fold_validation(model, x, y)
    return mean_acc, std_acc

def run_supervised_models(x, y):
    logistic_acc, std_acc = run_models(LogisticRegression(max_iter=10000), x, y)  # logistic regression
    print("Logistic Reg mean acc: ", logistic_acc, " standard dev of accuracies: ", std_acc)

    rf_acc, std_acc = run_models(RandomForestClassifier(), x, y)  # random forest classifier
    print(f"Random Forest Mean acc:", rf_acc, " standard dev of accuracies: ", std_acc)

    ada_acc, std_acc = run_models(AdaBoostClassifier(algorithm='SAMME'), x, y)  # ada boost classifier
    print(f"AdaBoost Mean acc:", ada_acc, " standard dev of accuracies: ", std_acc)

    bagging_acc, std_acc = run_models(BaggingClassifier(), x, y)  # bagging classifier
    print(f"Bagging Mean acc:", bagging_acc, " standard dev of accuracies: ", std_acc)

    xgb_acc, std_acc = run_models(XGBClassifier(enable_categorical=True), x, y)     # xg boost classifier
    print("XGBoost Mean acc:", xgb_acc, " standard dev of accuracies: ", std_acc)


def use_pca(x, y):
    # scale data before pca, x has been scaled already
    pca = PCA()
    scores = pca.fit_transform(x)
    pc1 = scores[:, 0]      # first principal component
    pc2 = scores[:, 1]      # second principal component

    colors = ['pink', 'purple']
    cmap = ListedColormap(colors)
    plt.scatter(pc1, pc2, c=y, cmap=cmap)   # plot pc1 versus pc2
    plt.title("PC 1 versus PC 2")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(["Autism", "Control"], colors)])
    plt.show()

    percentage_variance = pca.explained_variance_ratio_     # gives percentage variance that is explained by each component
    number_of_components = np.arange(pca.n_components_) + 1     # number of components in a list to use as X axis for the plot
    plt.plot(number_of_components, percentage_variance, color = "pink")
    plt.title("Percentage variance explained by all the principal components")
    plt.xlabel("Principal Components")
    plt.ylabel("Percentage variance explained")
    plt.show()
    cumulative_variance = pca.explained_variance_ratio_.cumsum()        # cumulates all the percentage variance by every component
    plt.plot(number_of_components, cumulative_variance, color = "pink")
    plt.title("Cumulative Percentage variance explained by all the principal components")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative percentage variance explained")
    plt.show()

    scores = pd.DataFrame(scores)
    run_supervised_models(scores, y)

def use_shap(x, y):
    model = XGBClassifier().fit(x, y)       # train xg boost model
    explainer = shap.Explainer(model, x)        # compute shap values
    shap_values = explainer(x)
    shap.plots.beeswarm(shap_values, max_display=30)    # choose top 30 features
    shap_features = [18159, 511, 7330, 20653, 15975, 10079, 13664, 2107, 8305, 11771, 1021, 19377, 8177, 7391, 15606, 1792, 4360, 17851, 5437, 14077, 2274, 9539, 16431, 9332, 17018, 20673, 856, 14147, 5315]
    x_shap = x.iloc[:, shap_features]
    run_supervised_models(x_shap, y)
    shap_columns = x.columns[shap_features]
    print(shap_columns)

def main():
    # save_df()     # Extract data from files and save in a directory in csv format
    df, x, y = load_data()       # Load data
    y = y.replace({'Control': False, 'ASD': True})
    # eda(df, x, y)       # INTERPRETATION: There are no null values, there exists a class imbalance but not to an extent where the model will be affected
    x = data_standardization(x)  # Standardize the gene expression values. Better accuracy (81%) when x is not standardized, otherwise (51%)
    # run_supervised_models(x, y)
    # use_pca(x, y)
    use_shap(x, y)

if __name__ == "__main__":
    main()

