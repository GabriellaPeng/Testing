from sys import platform

if platform == "darwin":
    root_path = '/Users/gabriellapeng/PycharmProjects/'
    test1_data = root_path + 'interview/data/'
    station_data = root_path + 'python/Exercise Files/Python Data Analysis/'
elif platform == "win32":
    test1_data = "E:\McGill\OneDrive - McGill University\life\selfimprovement\Skills\interview_prep_data\\"

