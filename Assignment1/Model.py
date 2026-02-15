import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import random




class Binclassification:
    def __init__(self,X,y,numcol,catcol):
        self.X = X
        self.y = y
        self.numcol = numcol
        self.catcol = catcol
        self.simulation_result_n={}
        self.simulation_result_one={}
        self.data_sim_one={}
        self.data_sim_n={}
        self.validate_and_assign_columns()
        
        
    def validate_and_assign_columns(self):
        numeric_cols=[]
        cat_cols=[]
        for Num,Cat,Col in zip(self.numcol,self.catcol,self.X.columns.tolist()):
            if(Num==1 and Cat==1):
                raise Exception("Column could not be both cat and num.")

            if(Num==1):
                numeric_cols.append(Col)
            elif(Cat==1):
                cat_cols.append(Col)
            else:
                raise Exception("A column has to be cat/numeric.")
        self.numcol=numeric_cols
        self.catcol=cat_cols


    
    
    
    def preprocess(self, EncodeCat=False,EncodeLabel=False,seed=42,sampling=2):
        
        '''
        other logic would be apply one hot and std scaler the in last apply smote
        then it would have 1 final pipline 
        
        '''

        '''
        Docstring for preprocess
        
        :param EncodeCat: by default false
        :param EncodeLabel: by default false
        :param seed: 
        :param sampling: 0 for no sampling, 1 for undersampling, 2 for oversampling
        '''
        
        # label encoding
        if EncodeLabel:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(self.y)
        else:
            y_encoded = self.y
        
        X_train,X_test,y_train,y_test=train_test_split(self.X,y_encoded,random_state=seed,test_size=0.3, stratify=y_encoded)

        
        # standardization
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        # numeric_processor = make_pipeline(
        #     SimpleImputer(strategy="mean"),
        #     StandardScaler()
        #     )
        
        scale_transformer = ColumnTransformer(
            transformers=[
            ("num", numeric_pipeline, self.numcol)
            ],
            remainder="passthrough"   # keep categorical untouched
            )
        
        X_train_scaled = scale_transformer.fit_transform(X_train)
        X_test_scaled  = scale_transformer.transform(X_test)
        
        # smotenc
        original_cols = list(self.X.columns)
        remainder_cols = [col for col in original_cols if col not in self.numcol]

        cat_indices = [
            len(self.numcol) + remainder_cols.index(cat_col)
            for cat_col in self.catcol
        ]

        if sampling==0:
            
            X_train_res,y_train_res=X_train_scaled,y_train
            
        elif sampling==1:

            rus = RandomUnderSampler(sampling_strategy='majority')
            X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train)
            
        elif sampling==2:
            
            smote = SMOTENC(
                categorical_features=cat_indices
                )
            X_train_res, y_train_res = smote.fit_resample(
                X_train_scaled, y_train)
            
        else:
            raise ValueError("invalid sampling input")
        
        
        print('After Sampling train_X: {}'.format(X_train_res.shape))
        print('After Sampling train_y: {} \n'.format(y_train_res.shape))

        
        # numeric_processor=Pipeline(
        #     steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
        #            ("scaler",StandardScaler())]
        # )

        
        
        
        # one hot 
        cat_processor=make_pipeline(
            OneHotEncoder(handle_unknown="ignore")            
        )
        cat_transformer = cat_processor if EncodeCat else "passthrough"
        
        Cat_transformation= ColumnTransformer([
            ('cat', cat_transformer, cat_indices)],
            remainder='passthrough')
        
        x_train_final = Cat_transformation.fit_transform(X_train_res)
        x_test_final  = Cat_transformation.transform(X_test_scaled)
        
        return x_train_final,x_test_final,y_train_res,y_test
    
    
    
    def _run_simulation(self, EncodeCat, EncodeLabel, seed, sampling):
        X_train, X_test, y_train, y_test = self.preprocess(EncodeCat,EncodeLabel,seed,sampling)
        
        results = {}

        models = {
            "LogisticRegression": (
                LogisticRegression(max_iter=5000),
                {
                    'C': np.logspace(-4, 4, 20),
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                }
            ),
            "SVC": (
                SVC(),
                {
                    'C': [0.1, 1, 10],
                    'gamma': [0.01, 0.1, 1],
                    'kernel': ['linear', 'rbf']
                }
            ),
            "DecisionTree": (
                DecisionTreeClassifier(),
                {
                    'max_depth': range(1, 15),
                    'min_samples_leaf': range(1, 20, 2),
                    'min_samples_split': range(2, 20, 2),
                    'criterion': ["entropy", "gini"]
                }
            )
        }


        for name, (model, params) in models.items():
            print("inside the loop 1")
            grid = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=StratifiedKFold(10),
                scoring='accuracy',
                n_jobs=-1,
                verbose=True
            )

            grid.fit(X_train, y_train)

            results[name] = {
                # "best_estimator": grid.best_estimator_,
                "best_score": grid.best_score_,
                "best_params": grid.best_params_
            }
            
        data = {
        "X_train":X_train,
        "X_test":X_test,
        "y_train":y_train,
        "y_test":y_test
        }
        return data, results




    def simulate_one(self, EncodeCat=False, EncodeLabel=False, seed=42, sampling=2):

        data, results = self._run_simulation(
            EncodeCat, EncodeLabel, seed, sampling
        )

        self.data_sim_one[f"seed {seed}"] = data
        self.simulation_result_one[f"seed {seed}"] = results

        # return results
        
            


    
    def simulate_ntimes(self, n=5, EncodeCat=False, EncodeLabel=False, sampling=2):

        seeds = random.sample(range(1, 101), n)

        for seed in seeds:

            data, results = self._run_simulation(
                EncodeCat, EncodeLabel, seed, sampling
            )

            self.data_sim_n[f"seed {seed}"] = data
            self.simulation_result_n[f"seed {seed}"] = results

        # return self.simulation_result_n

    # def preprocess(self, EncodeCat=False,EncodeLabel=False,seed=42,sampling=2):

    #     '''
    #     Docstring for preprocess
        
    #     :param EncodeCat: by default false
    #     :param EncodeLabel: by default false
    #     :param seed: 
    #     :param sampling: 0 for no sampling, 1 for undersampling, 2 for oversampling
    #     '''
        
        
        
    #     X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,random_state=seed,test_size=0.3, stratify=self.y)
        
    #     numeric_pipeline = Pipeline(steps=[
    #         ("imputer", SimpleImputer(strategy="mean")),
    #         ("scaler", StandardScaler())
    #     ])
    #     # numeric_processor = make_pipeline(
    #     #     SimpleImputer(strategy="mean"),
    #     #     StandardScaler()
    #     #     )
    #     scale_transformer = ColumnTransformer(
    #         transformers=[
    #         ("num", numeric_pipeline, self.numcol)
    #         ],
    #         remainder="passthrough"   # keep categorical untouched
    #         )
        
    #     X_train_scaled = scale_transformer.fit_transform(X_train)
    #     X_test_scaled  = scale_transformer.transform(X_test)

    #     if sampling==0:
            
    #         X_train_res,y_train_res=X_train,y_train
            
    #     if sampling==1:
            
    #         rus = RandomUnderSampler(sampling_strategy='majority')
    #         X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train)
            
    #     elif sampling==2:
    #         cat_indices = self.X.columns.get_indexer(self.catcol)
    #         smote = SMOTENC(
    #             categorical_features=cat_indices
    #             )
    #         X_train_res, y_train_res = smote.fit_resample(
    #             X_train_scaled, y_train)
            
    #     else:
    #         raise "invalid sampling input"
        
        
    #     print('After Sampling train_X: {}'.format(X_train_res.shape))
    #     print('After Sampling train_y: {} \n'.format(y_train_res.shape))

        
    #     # numeric_processor=Pipeline(
    #     #     steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
    #     #            ("scaler",StandardScaler())]
    #     # )

        

    #     cat_processor=make_pipeline(
    #         OneHotEncoder(handle_unknown="ignore")            
    #     )
    #     cat_transformer = cat_processor if EncodeCat else "passthrough"
        
    #     Cat_transformation= ColumnTransformer([
    #         ("num",numeric_pipeline,self.numcol)
    #         ('cat', cat_transformer, self.catcol)])
        
    #     x_train_final = Cat_transformation.fit_transform(X_train_res)
    #     x_test_final  = Cat_transformation.transform(X_test_scaled)
    #     return x_train_final,x_test_final,y_train_res,y_test
    
