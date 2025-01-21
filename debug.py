import datetime

birthday = datetime.datetime.strptime('2024-10-09', '%Y-%m-%d')
day100 = birthday + datetime.timedelta(days=100)
print(day100)
