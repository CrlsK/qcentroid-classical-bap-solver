import json
with open("input.json") as f:
    dic = json.load(f)
extra_arguments = dic.get("extra_arguments", {})
solver_params = dic.get("solver_params", {})
import qcentroid
result = qcentroid.run(dic["data"], solver_params, extra_arguments)
print(json.dumps(result, indent=2))
