import json

f = open('feature_extract.json', mode='w')
f.truncate()
f.close()
http_code_contract = [
    '400', '411', '505', '405', '417', '416', '200', '403', '501', '401',
    '404', '301', '500', '503'
]
content_list = [
    'Content-Type', 'Date', 'Connection', 'Content-Length', 'Server'
]
product_list = ['iis', 'nginx', 'http server']


def get_http_code(parts):
    http_code = list(parts.values())[0]['part_1'][9:12]
    if http_code in http_code_contract:
        return http_code
    else:
        return 'not_found'


def get_content_index(parts):
    index = [0] * len(content_list)
    current = 0
    part_2 = list(parts.values())[0]['part_2']
    for content in part_2:
        if content.split(':')[0] in content_list:
            index[current] = content_list.index(content.split(':')[0]) + 1
            current = current + 1
    return index


def test_http_code(http_data):
    http_code_list = {}
    for i in http_data:
        for parts in i["attribute"]:
            part_1 = list(parts.values())[0]['part_1'][9:12]
            part_1_str = list(parts.values())[0]['part_1']
            if part_1 not in list(http_code_list.keys()):
                http_code_list[part_1] = []
            if part_1_str not in http_code_list[part_1]:
                http_code_list[part_1].append(part_1_str)
    f = open('http_code.json', 'w')
    f.write(json.dumps(http_code_list, indent=1, ensure_ascii=False))
    f.write('\n')
    f.close()
    f = open('content.json', mode='w')
    f.write(
        json.dumps(
            {str(i + 1): content_list[i]
             for i in range(len(content_list))}))
    f.close()


def build_http_dict():
    return_dict = {}
    for http_code in http_code_contract:
        return_dict[http_code] = 0
    return_dict['part2_index'] = [0] * len(content_list)
    return_dict['body'] = 0
    for product in product_list:
        return_dict[product] = 0
    return return_dict


http_response_list = {}
count = 0
fo = open('processed_data.json', 'r')
while True:
    test_data_str = fo.readline()
    if test_data_str:
        count += 1
    else:
        break
    test_data = json.loads(test_data_str)
    attr_response_list = []
    for parts in test_data["attribute"]:
        http_dict = build_http_dict().copy()
        # 第一部分
        http_code = get_http_code(parts=parts)
        if http_code != 'not_found':
            http_dict[http_code] = 1
        # 第二部分
        http_dict['part2_index'] = get_content_index(parts).copy()
        # 第三部分
        if list(parts.values())[0]['part_3']:
            http_dict['body'] = 1
        http_dict[test_data['product']] = 1
        attr_response_list.append(http_dict)
    f = open('feature_extract.json', mode='a+')
    f.write(
        json.dumps({test_data["ip"]: attr_response_list}, ensure_ascii=False))
    f.write('\n')
    f.close()
fo.close()
