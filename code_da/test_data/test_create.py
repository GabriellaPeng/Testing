import time

from code_da.analysis_data.data_operations import df_insert_row
from code_da.get_data.load_data import load_text
from code_da.get_data.load_df import create_df
from code_da.prep_data.connect_sql import loadSql_to_df, loadDf_to_sql, \
    sql_engine_connect, fetch_query_data, execute_query
from path import root_path

data_path = root_path + "SQL/Ex_Files_Database_Clinic_MySQL/Chapter 05/"
character = data_path + 'characters.txt'
playlines = data_path + 'A_Midsummer_Nights_Dream.txt'

sqlEngine, dbConnection = sql_engine_connect(host='localhost', user='testuser',
                                             database='shakespeare', pw='temppass')

charList = load_text(character)
curChar = 'Unknown'

start_time = time.time()

amnd_colnames = ['line_number1', 'char_name1', 'play_text1']
df_amnd = create_df(amnd_colnames, name='new_amnd')

playlines = [line.strip() for line in open(playlines, 'r')]
line_num = 0

for line in playlines:
    if line.upper() in charList:
        # curChar = line.strip()
        print('Changing character to', line.strip())
    else:
        line_num = line_num + 1
        sql_values = line_num, curChar, line.strip()
        # print('Writing line \"' + line.strip() + '\"')
        df_insert_row(df_amnd, sql_values, ind=line_num - 1)
end_time = time.time()

loadDf_to_sql(df_amnd, dbConnection)
numPlayLines = fetch_query_data('SELECT COUNT(line_number1) FROM new_amnd;',
                                dbConnection)[0][0]
# indicate just take the number for the line_number
print(numPlayLines, 'rows')

queryExecTime = end_time - start_time
print('Total query time:', queryExecTime)

queryTimePerLine = queryExecTime / numPlayLines
print('Query time per line:', queryTimePerLine)

df = loadSql_to_df('SELECT * FROM new_amnd;',dbConnection)

df_performance = create_df(['query_type1', 'query_time1'], name='new_performance')
loadDf_to_sql(df_performance, dbConnection)

insertPerfSQL = f'INSERT INTO new_performance (query_type1, query_time1) VALUES (' \
                f'"CREATE", {queryTimePerLine});'

execute_query(insertPerfSQL, dbConnection)
dbConnection.close()