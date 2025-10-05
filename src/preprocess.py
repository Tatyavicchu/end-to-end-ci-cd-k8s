import os 
import yaml
import pandas as pd

root=os.path.dirname(os.path.dirname(__file__))
params_path=os.path.join(root,'params.yaml')
def load_params()->dict:
    try:
        with open(params_path)as f:
            params=yaml.safe_load(f)
            return params
    except:
        raise    
def load_data(params:dict)->pd.DataFrame:
    try:
        data_path=os.path.join(root,params["data"]["raw"])
        df=pd.read_csv(data_path)
        return df
    except:
        raise

def pre_process(df:pd.DataFrame,params:dict):
    try:
        df.dropna(inplace=True)
        data_path=os.path.join(root,params["data"]["preprocessed"])
        df.to_csv(data_path,index=False)
    except:
        raise

def main():
    params=load_params()
    df=load_data(params)
    pre_process(df,params)

if __name__=="__main__":
    main()