

def transfer_weights(c2f, c2f_v2,num_bottleneck):
    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()
    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight_cv1 = state_dict['cv1.conv.weight']
    old_weight_cv2 = state_dict['cv2.conv.weight']
    # print(f"Old cv1.conv.weight shape: {state_dict['cv1.conv.weight'].shape}")
    # print(f"New cv1.conv.weight shape: {state_dict_v2['cv1.conv.weight'].shape}")
    # print(f"Old cv2.conv.weight shape: {state_dict['cv2.conv.weight'].shape}")
    # print(f"New cv2.conv.weight shape: {state_dict_v2['cv2.conv.weight'].shape}")
    cv1_in,cv1_out = state_dict_v2['cv1.conv.weight'].shape[0],state_dict_v2['cv1.conv.weight'].shape[1]
    cv2_in,cv2_out = state_dict_v2['cv2.conv.weight'].shape[0],state_dict_v2['cv2.conv.weight'].shape[1]
    state_dict_v2['cv1.conv.weight'] = old_weight_cv1[:cv1_in,:cv1_out]
    state_dict_v2['cv2.conv.weight'] = old_weight_cv2[:cv2_in,:cv2_out]
    
    # print(f"Transferred cv1.conv.weight shape: {state_dict_v2['cv1.conv.weight'].shape}")
    # print(f"Transferred cv2.conv.weight shape: {state_dict_v2['cv2.conv.weight'].shape}")
    

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    bn1_shape = state_dict_v2['cv1.bn.weight'].shape[0]
    bn2_shape = state_dict_v2['cv2.bn.weight'].shape[0]
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn_v1 = state_dict[f'cv1.bn.{bn_key}']
        old_bn_v2 = state_dict[f'cv2.bn.{bn_key}']
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn_v1[:bn1_shape]
        state_dict_v2[f'cv2.bn.{bn_key}'] = old_bn_v2[:bn2_shape]
    

    for n in range(num_bottleneck):
        # Extract the weights from the old state dict for the Bottleneck layers
        old_weight_cv1 = state_dict[f'm.{n}.cv1.conv.weight']
        old_weight_cv2 = state_dict[f'm.{n}.cv2.conv.weight']

        # Get the new shapes for the Bottleneck layers in state_dict_v2
        cv1_in, cv1_out = state_dict_v2[f'm.{n}.cv1.conv.weight'].shape[0], state_dict_v2[f'm.{n}.cv1.conv.weight'].shape[1]
        cv2_in, cv2_out = state_dict_v2[f'm.{n}.cv2.conv.weight'].shape[0], state_dict_v2[f'm.{n}.cv2.conv.weight'].shape[1]

        # Transfer the weights for the Bottleneck layers
        state_dict_v2[f'm.{n}.cv1.conv.weight'] = old_weight_cv1[:cv1_in, :cv1_out]
        state_dict_v2[f'm.{n}.cv2.conv.weight'] = old_weight_cv2[:cv2_in, :cv2_out]

        # Transfer batchnorm weights and buffers for the Bottleneck layers
        bn1_shape = state_dict_v2[f'm.{n}.cv1.bn.weight'].shape[0]
        bn2_shape = state_dict_v2[f'm.{n}.cv2.bn.weight'].shape[0]

        for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
            old_bn_v1 = state_dict[f'm.{n}.cv1.bn.{bn_key}']
            old_bn_v2 = state_dict[f'm.{n}.cv2.bn.{bn_key}']
            state_dict_v2[f'm.{n}.cv1.bn.{bn_key}'] = old_bn_v1[:bn1_shape]
            state_dict_v2[f'm.{n}.cv2.bn.{bn_key}'] = old_bn_v2[:bn2_shape]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not (key.startswith('cv1.') or key.startswith('cv2.') or  key.startswith('m.')):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)
    c2f_v2.load_state_dict(state_dict_v2)
    print("C2f Weight transfer complete.")


def transfer_conv_weights(old_conv, new_conv):
    state_dict = old_conv.state_dict()
    state_dict_v2 = new_conv.state_dict()
    
    # Transfer conv weights
    old_weight_conv = state_dict['conv.weight']
    conv_in, conv_out = state_dict_v2['conv.weight'].shape[0], state_dict_v2['conv.weight'].shape[1]
    state_dict_v2['conv.weight'] = old_weight_conv[:conv_in, :conv_out, :, :]
    
    # Transfer batchnorm weights and buffers
    bn_shape = state_dict_v2['bn.weight'].shape[0]
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'bn.{bn_key}']
        state_dict_v2[f'bn.{bn_key}'] = old_bn[:bn_shape]
    
    for key in state_dict:
        if not (key.startswith('conv.') or key.startswith('bn.')):
            print(key)
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes    c2f_v.load_state_dict(state_dict_v2)

    for attr_name in dir(old_conv):
        attr_value = getattr(old_conv, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(new_conv, attr_name, attr_value)
    # Load the modified state_dict into the new model
    new_conv.load_state_dict(state_dict_v2)

    print("Conv Weight transfer complete.")
    
    
def transfer_sppf_weights(old_sppf, new_sppf):
    state_dict = old_sppf.state_dict()
    state_dict_v2 = new_sppf.state_dict()
    
    # Transfer weights for cv1
    old_weight_cv1 = state_dict['cv1.conv.weight']
    cv1_in, cv1_out = state_dict_v2['cv1.conv.weight'].shape[0], state_dict_v2['cv1.conv.weight'].shape[1]
    state_dict_v2['cv1.conv.weight'] = old_weight_cv1[:cv1_in, :cv1_out]
    
    # Transfer batchnorm weights and buffers for cv1
    bn_shape = state_dict_v2['cv1.bn.weight'].shape[0]
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[:bn_shape]
    
    # Transfer weights for cv2
    old_weight_cv2 = state_dict['cv2.conv.weight']
    cv2_in, cv2_out = state_dict_v2['cv2.conv.weight'].shape[0], state_dict_v2['cv2.conv.weight'].shape[1]
    state_dict_v2['cv2.conv.weight'] = old_weight_cv2[:cv2_in, :cv2_out]
    
    # Transfer batchnorm weights and buffers for cv2
    bn_shape = state_dict_v2['cv2.bn.weight'].shape[0]
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv2.bn.{bn_key}']
        state_dict_v2[f'cv2.bn.{bn_key}'] = old_bn[:bn_shape]
    
    for key in state_dict:
        if not (key.startswith('cv1.') or key.startswith('cv2.')):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(old_sppf):
        attr_value = getattr(old_sppf, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(new_sppf, attr_name, attr_value)
    
    # Load the modified state_dict into the new model
    new_sppf.load_state_dict(state_dict_v2)

    print("SPPF Weight transfer complete.")

def transfer_sequential_weights(old_seq, new_seq):
    state_dict_old = old_seq.state_dict()
    state_dict_new = new_seq.state_dict()

    for key in state_dict_new.keys():
        state_dict_new[key] = state_dict_old[key]
    for attr_name in dir(old_seq):
        attr_value = getattr(old_seq, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(new_seq, attr_name, attr_value)

    new_seq.load_state_dict(state_dict_new)
    print("Sequential weight transfer complete.")