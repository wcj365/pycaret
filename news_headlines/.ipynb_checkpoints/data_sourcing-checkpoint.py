#!/usr/bin/env python3

import pandas as pd
import json

def get_data():

    lst_dics = []

    with open('data/News_Category_Dataset_v2.json', mode='r', errors='ignore') as json_file:

        for dic in json_file:

            lst_dics.append( json.loads(dic) )

    ## print the first one

    print(lst_dics[0])

    df = pd.DataFrame(lst_dics)

    df = df[df["category"].isin(["POLITICS","ENTERTAINMENT", "TECH"])][["category","headline"]]
    print("Number of Rows:", df.shape[0])

    df.dropna(inplace=True)
    print("Number of Rows after removing rows with null values:", df.shape[0])

    df["headline"] = df["headline"].apply(lambda x: x.strip())
    df = df.query("headline != ''")
    print("Number of Rows after removing rows with blank headlines:", df.shape[0])
  
    df.to_csv("data/news_headlines.csv", index=False)
              
    return df


def main():
    
    get_data()
    
    
if __name__ == "__main__":
    
    main()