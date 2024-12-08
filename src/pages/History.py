import streamlit as st
import datetime

# Function to display history from recognition_history.txt

st.set_page_config(page_title="Recognition History", layout="wide")
st.title("Face Recognition History")

if st.button("Show Recognition History"):
    try:
        # Read data from recognition_history.txt
        with open("recognition_history.txt", "r") as file:
            history_data = file.readlines()

        if not history_data:
            st.write("No recognition history found.")

        # Display the history in a nice format
        st.markdown("### Recognition Log Entries")

        # Loop through each log entry and display it
        for entry in history_data:
            # Each entry in the log is in the format: "timestamp - name"
            timestamp, name = entry.strip().split(" - ")
            st.markdown(f"**Timestamp:** {timestamp}")
            st.markdown(f"**Recognized Name:** {name}")
            st.markdown("---")  # Horizontal line for separation

    except FileNotFoundError:
        st.error("`recognition_history.txt` file not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
