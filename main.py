import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def dataSplit(data,ratio):
    np.random.seed(42)
    shuffuled = np.random.permutation(len(data))
    test_size = int(len(data)*ratio)
    test_data = shuffuled[:test_size]
    train_data = shuffuled[test_size:]
    return data.iloc[train_data],data.iloc[test_data]

if __name__ =="__main__":
    df = pd.read_csv('datar2.csv')
    train ,test = dataSplit(df,0.2)
    x_train = train[['age','bmi','glucose','insulin','homa','leptin','adiponectin','resistin','mcp_1']].to_numpy()
    x_test = test[['age','bmi','glucose','insulin','homa','leptin','adiponectin','resistin','mcp_1']].to_numpy()
    y_train = train['classification'].to_numpy().reshape(504,-1)
    y_test = test[['classification']].to_numpy().reshape(125,-1)
    clf =LogisticRegression()
    clf.fit(x_train, y_train)
    prediction=clf.predict([[48,23.5,70,2.707,0.467408667,8.8071,9.7024,7.99585,417.114]])  #value change keli ki answer change honr No=1 and Yes=2
    print("RESULT:",prediction)
    print("No=1 and Yes=2")
    
    

    
    
