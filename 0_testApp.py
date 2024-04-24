import streamlit as st
import pandas as pd

def main():
    st.title("Leftover Ingredients Menu Agent AI")

    st.sidebar.header("Menu")
    selected_option = st.sidebar.radio("Select an option", ["Add Income", "Add Expense", "View Summary"])

    if selected_option == "Add Income":
        add_income()
    elif selected_option == "Add Expense":
        add_expense()
    elif selected_option == "View Summary":
        view_summary()

def add_income():
    st.header("Add Income")
    income_name = st.text_input("Income Name")
    income_amount = st.number_input("Amount", min_value=0.01, step=0.01)
    if st.button("Add Income"):
        with open("income.csv", "a") as f:
            f.write(f"{income_name},{income_amount}\n")
        st.success("Income added successfully!")

def add_expense():
    st.header("Add Expense")
    expense_name = st.text_input("Expense Name")
    expense_amount = st.number_input("Amount", min_value=0.01, step=0.01)
    if st.button("Add Expense"):
        with open("expenses.csv", "a") as f:
            f.write(f"{expense_name},{expense_amount}\n")
        st.success("Expense added successfully!")
        
def view_summary():
    st.header("Financial Summary")
    
    try:
        income_df = pd.read_csv("income.csv", names=["Name", "Amount"])
        expense_df = pd.read_csv("expenses.csv", names=["Name", "Amount"])

        total_income = income_df["Amount"].sum()
        total_expense = expense_df["Amount"].sum()
        balance = total_income - total_expense

        st.subheader("Income")
        st.dataframe(income_df)
        st.write(f"Total Income: ${total_income:.2f}")

        st.subheader("Expenses")
        st.dataframe(expense_df)
        st.write(f"Total Expenses: ${total_expense:.2f}")

        st.subheader("Balance")
        st.write(f"Balance: ${balance:.2f}")

    except FileNotFoundError:
        st.write("Data not found. No records available.")


if __name__ == "__main__":
    main()
