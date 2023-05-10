import yaml


def read_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result


def write_yaml(write_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data=write_data, stream=f, allow_unicode=True)

# dic = {
#     'labels':[1,2,3,4,5,6],
#     'name':'test',
#     'age':18
# }
# config_path = '/opt/projects/image_algorithm/config/config.yaml'
# r = read_yaml(config_path)
# print(r)
#
#
# r['labels'] = []
# path = '/opt/projects/image_algorithm/config/coco.yaml'
# res = read_yaml(path)
# print(res)
# # write_yaml(dic, './test.yaml')
#
# n = 0
# dic = {}
# for k,v in res.items():
#     dic[v] = {'index': n, 'id': k}
#     r['labels'].append(v)
#     n += 1
# print(dic)
# write_yaml(dic, '../../../config/label_ids.yaml')
# print(r)
# write_yaml(r, '../../../config/config.yaml')

