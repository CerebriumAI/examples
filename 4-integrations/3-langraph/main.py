
def run(param_1: str, param_2: str, run_id):  # run_id is optional, injected by Cerebrium at runtime
    my_results = {"1": param_1, "2": param_2}
    my_status_code = 200 # if you want to return a specific status code

    return {"my_result": my_results, "status_code": my_status_code} # return your results
    
# To deploy your app, run:
# cerebrium deploy
