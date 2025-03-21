import csv, re

# https://github.com/mechatroner/vscode_rainbow_csv/blob/master/rbql_core/README.md#rbql-rainbow-query-language-description

input_reader = csv.reader(open('data/gsm8k/test.csv'))
query = input('Please enter a query. Example: SELECT a1, a2 WHERE a2 != \'foobar\'\n> ') # select a1 where re.match(".*robe.*",a1.lower()) 
query = re.sub('(?<=[^a-zA-Z0-9_])a([0-9]+)', r'row[\1]', query) # Replaces a1 with row[1], a2 with row[2], etc
query = re.sub('SELECT', '', query, flags=re.IGNORECASE)
query_parts = re.split('WHERE', query, flags=re.IGNORECASE)
select_part = query_parts[0]
where_part = query_parts[1] if len(query_parts) > 1 else 'True'
main_loop = '''
for NR, row in enumerate(input_reader):
    row.insert(0, NR)
    if ({}):
        print(','.join(str(v) for v in [{}]))
'''.format(where_part, select_part)
exec(main_loop)


"""
rbql:
Supported SQL Keywords (Keywords are case insensitive)
SELECT
UPDATE
WHERE
ORDER BY ... [ DESC | ASC ]
[ LEFT | INNER ] JOIN
DISTINCT
GROUP BY
TOP N
LIMIT N
AS


我试过的例子
select len(a1) where re.match(".*robe.*",a1.lower()) and len(a1)<len(a2)
select len(a1) where re.match(".*robe.*",a1, re.INGORECASE) and len(a1)<len(a2)
select len(a1) where re.match(".*robe.*",a1) and len(a1)<len(a2)
select len(a1),a1 where re.match(".*robe.*",a1) order by len(a1) desc
select count(a1)
SELECT NR, * limit 10
"""