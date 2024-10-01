import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import os

def load_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = {}

    for f in csv_files:
        df = pd.read_csv(os.path.join(directory, f))

        if 'REMARKS' not in df:
            df['REMARKS'] = None

        dataframes[f] = df

    return dataframes

def remove_empty_rows(df):
    return df.dropna(how='all')

def calculate_row_numbers(df):
    current_idx = None
    row_number = -1  # Start at -1 so the first increment makes it 0
    row_numbers = []

    for index, row in df.iterrows():
        if pd.notna(row['IDX']):
            if row['IDX'] != current_idx:
                current_idx = row['IDX']
                row_number += 1  # Increment row_number only when a new IDX is encountered

            row_numbers.append(row_number)

        else:
            row_numbers.append(row_numbers[-1] if row_numbers else 0)

    df.loc[:, ['ROW_NUM']] = row_numbers

def calculate_match_status(df):
    df.loc[:, ['IS_MATCH']] = False

    for index, row in df.iterrows():
        current_row_num = row['ROW_NUM']
        exp_filters = df.loc[df['ROW_NUM'] == current_row_num, ['FILTERS_EXP', 'OPERATORS_EXP', 'VALUES_EXP']].apply(tuple, axis=1).tolist()
        act_tuple = (row['FILTERS_ACT'], row['OPERATORS_ACT'], row['VALUES_ACT'])

        if act_tuple in exp_filters:
            df.at[index, 'IS_MATCH'] = True

def display_aggrid(df):
    grid_options = GridOptionsBuilder.from_dataframe(df)
    grid_options.configure_default_column(sortable=False)
    grid_options.configure_column('MATCH', hide=True)
    grid_options.configure_column('ROW_NUM', hide=True)
    grid_options.configure_column('IS_MATCH', hide=True)
    grid_options.configure_column(field='REMARKS', editable=True)

    cell_style_jscode = JsCode('''
        function(params) {
            if (params.data.IS_MATCH) {
                return {
                    'color': 'black',
                    'backgroundColor': 'lightgreen'
                };
            } else {
                return {
                    'color': 'black',
                    'backgroundColor': 'lightcoral'
                };
            }
        }
    ''')

    for column in ['FILTERS_ACT', 'OPERATORS_ACT', 'VALUES_ACT']:
        grid_options.configure_column(column, cellStyle=cell_style_jscode)

    grid_options.configure_grid_options(domLayout='normal')
    grid_options = grid_options.build()
    
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        height=600,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
        key='df'
    )
    
    return grid_response['data']

def save_to_csv(dataframe, filepath):
    dataframe = dataframe.drop(columns=['ROW_NUM', 'IS_MATCH'], errors='ignore')
    dataframe.to_csv(filepath, index=False)

def csv_viewer_page(dataframes):
    # Sidebar to select CSV files
    st.sidebar.title('CSV File Selection')
    selected_files = st.sidebar.multiselect(
        'Select one or more CSV files to visualize:',
        list(dataframes.keys())
    )

    # Add a select box for filtering options
    filter_option = st.sidebar.selectbox(
        'Select filter option:',
        ('Default', 'Invalid Results', 'Valid Results')
    )

    st.title('CSV Viewer')

    if not selected_files:
        st.warning("Please select at least one CSV file.")
        return

    st.write(f'Visualizing file(s): {", ".join(selected_files)}')
    combined_df = pd.concat([dataframes[file] for file in selected_files], ignore_index=True)
    combined_df = remove_empty_rows(combined_df)
    calculate_row_numbers(combined_df)
    calculate_match_status(combined_df)

    # Apply filter based on the selected option
    if filter_option == 'Invalid Results':
        valid_row_numbers = combined_df[combined_df['IS_VALID'] == 'F']['ROW_NUM'].unique()

    elif filter_option == 'Valid Results':
        valid_row_numbers = combined_df[combined_df['IS_VALID'] == 'T']['ROW_NUM'].unique()

    else: 
        valid_row_numbers = combined_df['ROW_NUM'].unique()

    # Filter the DataFrame to keep all rows with the selected row numbers
    filtered_df = combined_df[combined_df['ROW_NUM'].isin(valid_row_numbers)]

    # Display the filtered DataFrame using AgGrid
    updated_df = display_aggrid(filtered_df)

    # Count invalid queries based on IS_VALID column
    total_invalid_entries = filtered_df['IS_VALID'].value_counts().get('F', 0)
    total_entries = filtered_df['IS_VALID'].count()

    # Display the count of invalid queries and total entries
    st.write(f'Total number of invalid queries: {total_invalid_entries} / {total_entries}')
    
    if st.button('Save changes'):
        save_to_csv(updated_df, os.path.join('dataset', 'output.csv'))
        st.success('Changes saved to output.csv')

    st.download_button(
        label="Download data as CSV",
        data=updated_df.drop(columns=['ROW_NUM', 'IS_MATCH'], errors='ignore').to_csv(index=False).encode('utf-8'),
        file_name='output.csv',
        mime='text/csv'
    )

def idx_viewer_page(dataframes):
    st.title('Invalid IDX Viewer')

    # Sidebar for selecting multiple CSV files
    st.sidebar.title('CSV File Selection')
    selected_files = st.sidebar.multiselect(
        'Select one or more CSV files:',
        list(dataframes.keys())
    )

    if not selected_files:
        st.warning("Please select at least one CSV file.")
        return

    # Combine data from selected files
    combined_df = pd.concat([dataframes[file] for file in selected_files], ignore_index=True)
    
    # Remove empty rows and calculate row numbers and match status
    combined_df = remove_empty_rows(combined_df)
    calculate_row_numbers(combined_df)
    calculate_match_status(combined_df)

    # Filter rows where IS_VALID is 'F'
    invalid_idx_df = combined_df[combined_df['IS_VALID'] == 'F']

    if invalid_idx_df.empty:
        st.write("No invalid IDX entries found.")
        return

    # Get unique IDX values with IS_VALID = 'F'
    unique_invalid_idx = invalid_idx_df['IDX'].dropna().unique().astype(int)

    # Dropdown to select IDX to view
    selected_idx = st.selectbox(
        'Select an invalid IDX to view:',
        unique_invalid_idx
    )

    if pd.notna(selected_idx):
        # Display all rows with the selected IDX, including associated rows with the same ROW_NUM
        selected_rows = combined_df[combined_df['ROW_NUM'].isin(
            invalid_idx_df[invalid_idx_df['IDX'] == selected_idx]['ROW_NUM'].unique()
        )]

        # Display the selected rows using AgGrid
        st.write(f"Displaying rows for IDX: {selected_idx}")
        updated_df = display_aggrid(selected_rows)

        # Calculate total occurrences of selected IDX and overall occurrences
        total_invalid_occurrences = selected_rows['IDX'].value_counts().get(selected_idx, 0)
        total_occurrences = combined_df['IDX'].value_counts().get(selected_idx, 0)

        # Display the count of occurrences
        st.write(f'Invalid IDX {selected_idx} appears: {total_invalid_occurrences} / {total_occurrences}')


        if st.button('Save changes'):
            save_to_csv(updated_df, os.path.join('dataset', 'output.csv'))
            st.success('Changes saved to output.csv')

        st.download_button(
            label="Download data as CSV",
            data=updated_df.drop(columns=['ROW_NUM', 'IS_MATCH'], errors='ignore').to_csv(index=False).encode('utf-8'),
            file_name='output.csv',
            mime='text/csv'
        )

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select a page:', ['CSV Viewer', 'Invalid IDX Viewer'])

    csv_directory = 'dataset'
    dataframes = load_csv_files(csv_directory)

    if page == 'CSV Viewer':
        csv_viewer_page(dataframes)

    elif page == 'Invalid IDX Viewer':
        idx_viewer_page(dataframes)

if __name__ == '__main__':
    main()