import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
import plotly.express as px

df = pd.read_csv('d:\DS\Programming\Fasih\Fasih_Project\weatherAUS.csv')

