# E-Commerce Sales EDA — Weekend Project
**ProSensia AI/ML Bootcamp | Weekend Project**

## Project Overview
Comprehensive Exploratory Data Analysis (EDA) on an E-Commerce Sales Dataset containing 10,000 orders across 26 features including sales, profit, shipping, customer segments, and regional data.

## Dataset
- **File:** `ecommerce_sales_dataset.csv`
- **Rows:** 10,000 orders
- **Columns:** 26 features
- **Years:** 2021–2024
- **Regions:** North America, Asia, Europe, Middle East

## Project Structure
```
├── ecommerce_eda.ipynb          # Main Jupyter Notebook (EDA)
├── ecommerce_sales_dataset.csv  # Raw dataset
├── ecommerce_cleaned.csv        # Cleaned & preprocessed dataset
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter Notebook
jupyter notebook ecommerce_eda.ipynb
```

## Key Business Insights

1. **Category Performance** — Electronics & Home/Kitchen generate the highest revenue; Beauty & Health has the best profit margins.
2. **Revenue Growth** — Clear upward revenue trend from 2021 to 2023; seasonal spikes visible in Q4.
3. **Customer Segments** — VIP and Premium customers drive disproportionate revenue despite lower order volume.
4. **Discount Risk** — Negative correlation between discount rate and profit margin. Discounts above 20% significantly erode profitability.
5. **Loss Orders** — Approximately 15% of all orders result in negative profit — requires pricing and cost structure review.
6. **Fulfilment Leakage** — Returns + Cancellations account for ~25% of orders, representing major revenue leakage.
7. **Shipping Efficiency** — Economy shipping is most used but has highest delivery time — risk to customer satisfaction.

## Visualizations Produced
- Correlation Heatmap
- Revenue & Profit by Category
- Annual & Monthly Revenue Trends
- Customer Segment Distribution
- Discount vs Profit Margin Analysis
- Regional Performance
- Order Status Distribution
- Distribution Plots (Revenue, Profit, Shipping Days, Profit Margin)

## Tools & Libraries
- Python 3.x
- Pandas — Data manipulation
- NumPy — Numerical operations
- Matplotlib — Visualizations
- Seaborn — Statistical plots
- Jupyter Notebook — Interactive analysis
