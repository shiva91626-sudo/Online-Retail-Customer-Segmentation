🛒 Online Retail Customer Segmentation (RFM + K-Means)

Unsupervised Machine Learning project using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to segment customers of the Online Retail Dataset (UCI).

📌 Project Overview

Customer segmentation helps businesses understand different customer groups and personalize marketing, improve retention, and increase revenue.

In this project, I used:

RFM Features

Recency — How recently a customer purchased

Frequency — How often they purchased

Monetary — Total amount they spent

K-Means Clustering (Unsupervised ML)

Elbow Method to find the optimal number of clusters

2D & 3D visualizations to understand customer behaviour

📂 Dataset

The dataset contains real e-commerce transactions from 2010–2011:

InvoiceNo

StockCode

Description

Quantity

InvoiceDate

UnitPrice

CustomerID

Country

🔧 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

🧮 RFM Feature Engineering
Recency

Days since last purchase:

ref_date = online_retail['InvoiceDate'].max() + dt.timedelta(days=1)
Recency = (ref_date - last_purchase_date).days

Frequency

Number of invoices per customer:

'InvoiceNo': 'count'

Monetary

Total spending:

'TotalPrice': 'sum'


We grouped customers using:

rfm = online_retail.groupby('CustomerID').agg({...})

🧬 Data Scaling

K-Means requires scaled data:

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

🔍 Choosing Optimal Clusters — Elbow Method

We tested K = 1 to 10 and plotted SSE (Inertia).

for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(rfm_scaled)
    sse.append(km.inertia_)


The elbow indicated K = 4.

🤖 K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(rfm_scaled)
rfm['Cluster'] = kmeans.labels_

📊 Cluster Summary
Cluster | Recency | Frequency | Monetary
--------|----------|-----------|----------
0       | Low R    | High F    | High M
1       | Mid R    | Low F     | Low M
2       | Very Low R | Very High F | Very High M
3       | High R   | Very Low F | Very Low M

Cluster Interpretation

Cluster 2 — Best Customers
Very loyal, frequent buyers, highest spending.

Cluster 0 — Loyal Customers
Buy regularly with healthy spending.

Cluster 1 — Occasional Customers
Infrequent but not inactive.

Cluster 3 — At-Risk / Lost Customers
Haven’t purchased recently, very low activity.

📈 Visualizations
Elbow Method

<img width="589" height="455" alt="elbow_plot" src="https://github.com/user-attachments/assets/c1f95104-a2b1-43df-9576-4b73809a0088" />


2D Cluster Plot (Recency vs Monetary)

<img width="597" height="455" alt="segment_2d" src="https://github.com/user-attachments/assets/3a649948-e9ad-4e99-ba14-fd7eba5df1c7" />


3D Cluster Plot (RFM)

<img width="513" height="512" alt="segment_3d" src="https://github.com/user-attachments/assets/788b7ddc-654f-4558-b2ae-504ee9d752fb" />



💡 Business Insights

A small group of customers contributes the majority of revenue.

Marketing should target Cluster 2 and Cluster 0 to maintain loyalty.

Cluster 3 needs re-activation campaigns.

Cluster 1 can be nurtured into loyal customers with offers/discounts.

🚀 How to Run
pip install -r requirements.txt
jupyter notebook


Run Online_Retail.ipynb.

⭐ Future Enhancements

Use Silhouette Score to validate clusters

Add PCA for dimensionality reduction

Try DBSCAN and Hierarchical Clustering

Build a dashboard (Power BI / Streamlit)

🙌 Author

Akash Raj
Data Analyst | Machine Learning Learner
