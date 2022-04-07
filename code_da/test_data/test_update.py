import time

from code_da.get_data.load_data import load_text
from path import root_path
from code_da.prep_data.connect_sql import sql_engine_connect, execute_query, \
    fetch_query_data

character = root_path + "SQL/Ex_Files_Database_Clinic_MySQL/Chapter 05/" + 'characters.txt'

sqlEngine, dbConnection = sql_engine_connect(host='localhost', user='testuser',
                                             database='shakespeare', pw='temppass')

start_time = time.time()
updateSQL = 'UPDATE new_amnd SET play_text1 = REPLACE(play_text1, %s, %s);'

for character in load_text(character):
    print(f'Capitalizing occurances of {character}...')
    updateStrings = f'"{character.capitalize()}"', f'"{character.upper()}"'
    query = updateSQL % updateStrings
    execute_query(query, dbConnection)
# with open(character,'r') as char:
# 	for character in char.read().splitlines():
# 		print('Capitalizing occurances of ' + character + '...')
# 		updateStrings = character.capitalize(), character.upper()
# 		cur.execute(updateSQL, updateStrings)

end_time = time.time()

numPlayLines = fetch_query_data('SELECT COUNT(line_number1) FROM new_amnd;',
								dbConnection)[0][0]
print(numPlayLines, 'rows')

queryExecTime = end_time - start_time
print('Total query time:', queryExecTime)
queryTimePerLine = queryExecTime / numPlayLines
print('Query time per line:', queryTimePerLine)

insertPerfSQL = f'INSERT INTO new_performance (query_type1, query_time1) VALUES (' \
				f'"UPDATE", {queryTimePerLine});'
execute_query(insertPerfSQL, dbConnection)

dbConnection.close()