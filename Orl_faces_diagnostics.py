import pandas as pd

orl_keypoins_path = "C:/Users/Michal/Documents/magisterka/orl_faces_keypoints.csv"

orl_keypoints = pd.DataFrame.from_csv(orl_keypoins_path)

print(list(orl_keypoints.columns.values))