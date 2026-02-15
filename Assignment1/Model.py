import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class Binclassification:
    def __init__(self):
        self.X
        self.y
        self.numcol
        self.catcol
        self.simulation_result
        self.simulate_one_data
        self.simulate_n_data
        

    
    def preprocess(self, EncodeCat=False,EncodeLabel=False,seed=42,sampling=1):
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
        
        numeric_processor=Pipeline(
            steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
                   ("scaler",StandardScaler())]
        )
        # numeric_processor = make_pipeline(
        #     SimpleImputer(strategy="mean"),
        #     StandardScaler()
        #     )

        cat_processor=make_pipeline(
            OneHotEncoder(handle_unknown="ignore")            
        )
        cat_transformer = cat_processor if EncodeCat else "passthrough"

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', cat_transformer, cat_cols)])
        
        x_train_std = preprocessor.fit_transform(x_train)
        x_test_std  = preprocessor.transform(x_test)
    
    def simulate():
        pass
    def simulate_ntimes():
        pass