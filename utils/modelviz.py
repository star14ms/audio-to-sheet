import os


def profile_model(model, inputs):
    from thop import profile
    macs, params = profile(model, inputs=inputs, verbose=False)
    print(model)
    print('모델 생성 완료! (MACs: {} G | Params: {} M)'.format(
        round(macs/1000/1000/1000, 2), 
        round(params/1000/1000, 2),
    ))


def draw_graphs(model, inputs, min_depth=1, max_depth=10, directory='./model_viz/', hide_module_functions=True, input_names=None, output_names=None):
    from torchview import draw_graph
    base_filename = directory + model.__class__.__name__ + '.'
    input_names = input_names if isinstance(input_names, (tuple, list)) else (input_names,)
    output_names = output_names if isinstance(output_names, (tuple, list)) else (output_names,)

    for input_1, input_name in zip(inputs, input_names):
        input_1.label = input_name

    for i in range(min_depth, max_depth+1):
        draw_graph(
            model, 
            input_data=inputs,
            expand_nested=True, 
            depth=i, 
            save_graph=True, 
            graph_name=model.__class__.__name__ + '.' + f'{i}_depth',
            directory=directory,
            hide_module_functions=hide_module_functions,
            output_names=output_names
        )
        
        current_file = base_filename + f'{i}_depth' + '.gv'

        if i > min_depth:
            previous_file = base_filename + f'{i-1}_depth' + '.gv'

            # open previous saved .gv file and check text is same as current one
            with open(current_file) as file:
                data_pre = file.read()
            with open(previous_file) as file:
                data_cur = file.read()

            if len(data_pre) == len(data_cur):
                # remove current file
                os.remove(current_file)
                os.remove(current_file + '.png')
                break

        print(f'Graph for {i} depth is saved at {current_file}')
