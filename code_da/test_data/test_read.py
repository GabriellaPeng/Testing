import time

from code_da.prep_data.connect_sql import sql_engine_connect, fetch_query_data, \
    execute_query

sqlEngine, dbConnection = sql_engine_connect(host='localhost', user='testuser',
                                             database='shakespeare', pw='temppass')

start_time = time.time()

for i in fetch_query_data('SELECT play_text1 FROM new_amnd;', dbConnection):
    print(i[0])
# for line in cur.fetchall():
#     print(line[0])

end_time = time.time()

numPlayLines = fetch_query_data('SELECT count(line_number) FROM amnd;', dbConnection)[0][0]
print(numPlayLines,'rows')

queryExecTime = end_time - start_time
print('Total query time:',queryExecTime)
queryTimePerLine = queryExecTime / numPlayLines
print('Query time per line:',queryTimePerLine)

insertPerfSQL = f'INSERT INTO new_performance (query_type1, query_time1) VALUES (' \
                f'"READ", {queryTimePerLine});'
execute_query(insertPerfSQL, dbConnection)

dbConnection.close()