# German Credit Data Explorer

An interactive Streamlit web application to explore, visualize, and analyze the **German Credit Dataset**.  
This project provides dynamic filtering, charting, and statistical summaries — all through an easy-to-use interface.

## 📂 Project Structure
- **Data Loading:** Loads and caches the processed German Credit data.
- **Interactive Filtering:** Search and filter rows based on keywords.
- **Custom Column Display:** Select which columns to display dynamically.
- **Visualizations:**
  - Bar charts for categorical features.
  - Line charts for numeric features.
  - Correlation matrix for numerical columns.
- **Statistics:** Expandable section showing dataset summary (`mean`, `std`, etc.).

## 🚀 Features
- Sidebar for controls
- Dynamic filtering by text input
- Select columns to view and visualize
- Automatic bar charts for categorical columns
- Line charts for numerical columns
- Correlation table between numerical features
- Expandable summary statistics section

## 🛠️ Built With
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Python 3.x](https://www.python.org/)

## 📊 Screenshots
| Feature | Preview |
|:--------|:--------|
| Column Selection | ![image](https://github.com/user-attachments/assets/d924ea11-211b-4e98-9469-918e358a0a01)|
| Bar Chart | ![image](https://github.com/user-attachments/assets/99676ca3-7f92-46e0-bff5-1a493376a109)|
| Line Chart |![image](https://github.com/user-attachments/assets/d08b8a3f-6707-4d16-ae7b-0692a88a34c2)|
| Correlation Table |![image](https://github.com/user-attachments/assets/d796b934-053e-4bb1-8a3f-3c7df4f6ac97)|

> (📸 *Tip: After deploying your app, replace the `#` with actual image links!*)

## 📄 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/credit-data-explorer.git
   cd credit-data-explorer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📝 Requirements

Create a `requirements.txt` file with:

```
streamlit
pandas
```

(You can extend if you add more libraries.)

## 💡 Future Improvements
- Add a heatmap for correlation visualization
- Include advanced filtering options
- Deploy the app online (Streamlit Community Cloud / Hugging Face Spaces)

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
This project is open source under the [MIT License](LICENSE).

---

# 🚀 Quick Start

```bash
pip install streamlit pandas
streamlit run app.py
```

---
> *Built with ❤️ using Streamlit and Python.*
