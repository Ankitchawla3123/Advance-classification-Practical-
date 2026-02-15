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


class Binclassification:
    def __init__(self,X,y,numcol,catcol):
        self.X = X
        self.y = y
        self.numcol = numcol
        self.catcol = catcol
        self.simulation_result=None
        self.data_sim_one=None
        self.data_sim_n=None
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



    def simulate(self,EncodeCat=False,EncodeLabel=False,seed=42,sampling=2):
        X_train, X_test, y_train, y_test=self.preprocess(EncodeCat,EncodeLabel,seed,sampling)
        self.data_sim_one={
            f'seed {seed}':{
                "X_train":X_train,
                "X_test":X_test,
                "y_train":y_train,
                "y_test":y_test
            }
        }
        
    def simulate_ntimes(self,n=5,EncodeCat=False,EncodeLabel=False,seed=42,sampling=2):
        
        

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
    
