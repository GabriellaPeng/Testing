import time
from code_da.prep_data.connect_sql import sql_engine_connect, fetch_query_data, \
    execute_query

sqlEngine, dbConnection = sql_engine_connect(host='localhost', user='testuser',
                                             database='shakespeare', pw='temppass')

start_time = time.time()
numPlayLinesBeforeDelete = \
    fetch_query_data('SELECT COUNT(line_number1) FROM new_amnd;', dbConnection
                     )[0][0]
print('Deleting lines...')

execute_query('DELETE FROM new_amnd WHERE play_text1 RLIKE '
              '"^enter|^exit|^act|^scene|^exeunt";', dbConnection)
end_time = time.time()

numPlayLinesAfterDelete = fetch_query_data('SELECT COUNT(line_number1) FROM new_amnd;',
                                           dbConnection)[0][0]

numPlayLinesDeleted = numPlayLinesBeforeDelete - numPlayLinesAfterDelete
print(
    f'deleted {numPlayLinesDeleted} rows, from {numPlayLinesBeforeDelete} to'
    f' {numPlayLinesAfterDelete}')

queryExecTime = end_time - start_time
print('Total query time:', queryExecTime)

queryTimePerLine = queryExecTime / numPlayLinesDeleted
print('Query time per line:', queryTimePerLine)

insertPerfSQL = f'INSERT INTO new_performance (query_type1, query_time1) VALUES (' \
                f'"DELETE", {queryTimePerLine});'
execute_query(insertPerfSQL, dbConnection)

dbConnection.close()
