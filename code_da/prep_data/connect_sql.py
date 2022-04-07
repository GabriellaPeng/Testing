from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.engine.url import URL


# def sql_connect(host='localhost', user='admin', password='Gaby100!', database='accidents',
#                 ifcommmit=False):
#     # set up the db for mysql
#     myConnection = pymysql.connect(host=host, user=user, passwd=password, db=database)
#     cur = myConnection.cursor()
#
#     if ifcommmit:
#         myConnection.commit()
#         myConnection.close()
#
#     return cur
#
#
# def executeSql(query):
#     cur = sql_connect()
#     # run mysql
#     cur.execute(query)  # 'SELECT vtype From vehicle_type WHERE vtype LIKE "%otorcycle%";'
#     # get the result out need to assign to a variable
#     data = cur.fetchall()
#     return data

def sql_engine_connect(host, user, database, pw):
    sqlEngine = create_engine(URL(
        drivername='mysql+pymysql', username=user, password=pw, host=host,
        database=database
    ))

    dbConnection = sqlEngine.connect()
    return sqlEngine, dbConnection


def execute_query(sql_query, dbConnection):
    return dbConnection.execute(sql_query)


def loadSql_to_df(sql_query, dbConnection):
    """Returns (table, connection). table is a pandas DataFrame."""
    frame = pd.read_sql(sql_query, con=dbConnection)
    pd.set_option('display.expand_frame_repr', False)

    # dbConnection.close()
    return frame


def loadDf_to_sql(dataFrame, dbConnection, closeConn=False):
    '''Write Pandas DataFrame into a MySQL Database Table'''
    if not dataFrame.name:
        dataFrame.name = str([x for x in globals() if globals()[x] is dataFrame][0])
    table_name = dataFrame.name

    try:
        frame = dataFrame.to_sql(name=table_name, con=dbConnection, if_exists='fail')
    except ValueError as vx:
        print(vx)
    except Exception as ex:
        print(ex)
    else:
        print(f"Table {table_name} created successfully.")
    if closeConn:
        dbConnection.close()


def fetch_query_data(sql_query, dbConnection):
    fetch_data = dbConnection.execute(sql_query).cursor.fetchall()
    # dbConnection.close()
    return fetch_data


def read_sqlquery_to_pd(query, sqlEngine):
    return pd.read_sql_query(query, sqlEngine)


def create_sql_table(colname_types, table_name, dbConnection):
    '''
    Create SQL table
    :param colname_types: {id: INT NOTNULL AUTO_INCREMENT PRIMARY KEY, name: VARCHAR(
    10) NULL,
    address: TEXT, time: FLOAT, age: INT}
    '''
    colnames = list(colname_types.keys())
    query = f'CREATE TABLE {table_name} ('

    for colname, coltype in colname_types.items():
        query += f'"{str(colnames)}" {str(coltype)},'

    query += ')'

    execute_query(query, dbConnection)


'''
selectSQL = ('
    SELECT t.vtype, a.accident_severity
    FROM accidents_2015 AS a
    JOIN vehicles_2015 AS v ON a.accident_index = v.accident_index 
    JOIN vehicle_type AS t ON t.vcode = v.vehicle_type
    WHERE t.vtype LIKE %s
    ORDER BY a.accident_severity; ')
    
    insertSQL = ('INSERT INTO accident_medians VALUES (%s, %s); ')
    
    for cycle in cycleList:
        cur.execute(selectSQL, cycle[0])
        accidents = cur.fetchall()
        quotient, remainder = divmod(len(accidents), 2)
        if remainder:
            med_sev = accidents[quotient][1]
        else:
            med_sev = (accidents[quotient][1] + accidents[quotient + 2][1]) / 2
        print('Finding median for', cycle[0], '\n', f'median is {med_sev}')
        cur.execute(insertSQL, (cycle[0], med_sev))
    
'''
