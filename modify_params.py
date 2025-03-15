from ruamel.yaml import YAML

# 创建 YAML 对象，设置保留注释和格式
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

# 读取 YAML 文件
with open(r'./params.yml', 'r', encoding='utf-8') as file:
    params = yaml.load(file)

# 修改 design_name
new_design_name = 'multiplier'
# new_design_name = 'adder'
params['design_name'] = new_design_name

# 同步修改 design_file 和 playground_dir
params['design_file'] = f'design/{new_design_name}.v'
params['playground_dir'] = f'playground/{new_design_name}'

# 将修改后的内容写回文件
with open(r'./params.yml', 'w', encoding='utf-8') as file:
    yaml.dump(params, file)

print('Completed')