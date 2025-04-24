def process_through_global_workspace_(
    global_workspace: Any,
    preprocessed_samples: List[dict],
    device: torch.device,
    debug = False,
) -> List[dict]:
    """
    Processes samples through the Global Workspace.
    
    Args:
        global_workspace: The global workspace model.
        preprocessed_samples: List of preprocessed samples.
        device: Torch device.
        debug: If True, generates random outputs instead of processing through model.
        
    Returns:
        List of processed samples with decoded images and shape colors.
    """
    processed_results = []
    
    for sample in tqdm(preprocessed_samples, desc="Translating samples through Global Workspace"):
        if debug:
            # Generate random outputs for debugging
            decoded_image = np.random.random((32, 32, 3))  # Random image with size 32x32x3
            mask = np.random.random((32, 32)) > 0.5  # Random boolean mask
            mean_color_shape = np.random.randint(0, 256, size=3)  # Random RGB color
            
            processed_results.append({
                'original_sample': sample,
                'decoded_image': decoded_image,
                'mask': mask,
                'shape_color': mean_color_shape
            })
            continue
            
        one_hot = sample['model_inputs']['one_hot']
        attr_vector = sample['model_inputs']['attr_vector']
        
        gw_inputs = [one_hot, attr_vector]
        
        attr_gw_latent = global_workspace.gw_mod.encode(
            {"attr": global_workspace.encode_domain(gw_inputs, "attr")}
        )
        gw_latent = global_workspace.gw_mod.fuse(
            attr_gw_latent,
            {"attr": torch.ones(attr_gw_latent["attr"].size(0)).to(device)}
        )
        

        #Translate through the global workspace
        decoded_latents = global_workspace.gw_mod.decode(gw_latent)["v_latents"]
        decoded_images_tensor = global_workspace.domain_mods["v_latents"].decode_images(decoded_latents)[0]
        decoded_image = decoded_images_tensor.permute(1, 2, 0).detach().cpu().numpy() 
        mask = segment_shape(decoded_image)
        mean_color_shape = extract_shape_color(decoded_image, mask)
        mean_color_shape = mean_color_shape * 255 

        # Half cycle through the global workspace
        decoded_half_latents = global_workspace.gw_mod.decode(gw_latent)["attr"]
        

        # Full cycle through the global workspace
        vlat_gw_latent = global_workspace.gw_mod.encode(
            {"v_latents": global_workspace.encode_domain(decoded_latents, "v_latents")}
        )
        gw_latent = global_workspace.gw_mod.fuse(
            vlat_gw_latent,
            {"v_latents": torch.ones(vlat_gw_latent["v_latents"].size(0)).to(device)}
        )
        cycle_decoded_attr_tensor = global_workspace.gw_mod.decode(gw_latent)["attr"]
        
        # TODO renvoyer des valeurs sofmaxiser plutot pour les catégories

        # get colors ie last three values of the tensors 
        colors_half = decoded_half_latents[0][-3:]
        colors_cycle = cycle_decoded_attr_tensor[0][-3:]
       
        processed_results.append({
            'original_sample': sample,
            'decoded_image': decoded_image,
            'mask': mask,
            'shape_color': mean_color_shape, # RGB color
            'half_cycle': colors_half,
            'full_cycle': colors_cycle
        })
        
    return processed_results

from torchvision.transforms.functional import to_tensor
def process_through_global_workspace_batch(
    global_workspace: Any,
    preprocessed_samples: List[Dict],
    device: torch.device,
    debug: bool = False,
    batch_size: int = 2056,  # Process samples in batches
) -> List[Dict]:
    """
    Processes samples through the Global Workspace, focusing on visual transformations.
    Calculates:
    1. Translation: Attribute -> GW -> Visual Image
    2. Half-Cycle: Visual -> GW -> Visual Image
    3. Full-Cycle: Visual -> GW -> Attribute -> GW -> Visual Image

    Args:
        global_workspace: The global workspace model instance.
        preprocessed_samples: List of preprocessed samples. Each sample dict
                              is expected to contain sample['model_inputs']
                              with keys like 'attr_vector', 'one_hot', and 'v_latent'.
        device: Torch device.
        debug: If True, generates random outputs instead of processing through model.
        batch_size: Number of samples to process in a batch.

    Returns:
        List of processed samples, each containing the original sample info
        and the decoded images (and their extracted colors) for the three paths.
    """
    processed_results = []
    global_workspace.eval()  # Make sure model is in evaluation mode

    # Debug mode - fast random output generation
    if debug:
        random_image = lambda: np.random.random((32, 32, 3))
        random_mask = lambda: np.random.random((32, 32)) > 0.5
        random_color = lambda: np.random.randint(0, 256, size=3)
        
        for sample in preprocessed_samples:
            processed_results.append({
                'original_sample': sample,
                'translated_image': random_image(),
                'translated_mask': random_mask(),
                'translated_shape_color': random_color(),
                'half_cycle_image': random_image(),
                'half_cycle_mask': random_mask(),
                'half_cycle_shape_color': random_color(),
                'full_cycle_image': random_image(),
                'full_cycle_mask': random_mask(),
                'full_cycle_shape_color': random_color(),
            })
        return processed_results

    # Process samples in batches for better GPU utilization
    for i in range(0, len(preprocessed_samples), batch_size):
        batch = preprocessed_samples[i:i+batch_size]
        
        with torch.no_grad():  # Disable gradient calculations for inference
            # Pre-process batch data
            one_hot_batch = []
            attr_vector_batch = []
            v_latent_vectors = []
            visual_ground_truth_tensors = []
            
            # Collect batch inputs
            for sample in batch:
                # Attribute inputs
                one_hot = sample['model_inputs']['one_hot'].to(device)
                attr_vector = sample['model_inputs']['attr_vector'].to(device)
                one_hot_batch.append(one_hot)
                attr_vector_batch.append(attr_vector)
                
                # Visual inputs
                if 'visual_ground_truth' not in sample:
                    raise ValueError(f"Sample missing 'visual_ground_truth'. Sample keys: {sample.keys()}")
                
                img = sample["visual_ground_truth"]
                # Convert to tensor and prepare for model
                visual_gt_tensor = to_tensor(img)[:3].unsqueeze(0).to(device)
                visual_ground_truth_tensors.append(visual_gt_tensor)
                
                # Encode visual inputs once to avoid redundant processing
                visual_module = global_workspace.domain_mods["v_latents"]
                v_latent = visual_module.visual_module.encode(visual_gt_tensor)
                v_latent_vectors.append(v_latent)
            
            # Stack batch tensors where possible
            one_hot_batch = torch.cat(one_hot_batch, dim=0) if len(one_hot_batch) > 1 else one_hot_batch[0]
            attr_vector_batch = torch.cat(attr_vector_batch, dim=0) if len(attr_vector_batch) > 1 else attr_vector_batch[0]
            
            # --- BATCH PROCESSING: Path 1 (Attribute -> GW -> Visual) ---
            # Process attribute inputs as a batch when possible
            attr_inputs = [one_hot_batch, attr_vector_batch]
            attr_domain_batch = global_workspace.encode_domain(attr_inputs, "attr")
            attr_gw_latent = global_workspace.gw_mod.encode({"attr": attr_domain_batch})
            
            # Create presence mask for the batch
            presence_mask = torch.ones(attr_gw_latent["attr"].size(0)).to(device)
            gw_latent_from_attr = global_workspace.gw_mod.fuse(attr_gw_latent, {"attr": presence_mask})
            
            # Decode GW -> Visual Latents
            translated_v_latents = global_workspace.gw_mod.decode(gw_latent_from_attr)["v_latents"]
            
            # Decode Visual Latents -> Images (potentially in batch)
            translated_images = global_workspace.domain_mods["v_latents"].decode_images(translated_v_latents)
            
            # --- PROCESS EACH SAMPLE INDIVIDUALLY FOR PATHS 2 & 3 ---
            for idx, sample in enumerate(batch):
                # Get the appropriate tensors for this sample
                v_latent_vector = v_latent_vectors[idx]
                
                # Process translated image for this sample
                translated_image_tensor = translated_images[idx] if isinstance(translated_images, list) else translated_images[0]
                translated_image_np = translated_image_tensor.permute(1, 2, 0).detach().cpu().numpy()
                
                # Use threading or multiprocessing for CPU-bound operations
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit segmentation and color extraction tasks
                    translated_mask_future = executor.submit(segment_shape, translated_image_np)
                    
                    # --- Path 2: Half-Cycle (Visual -> GW -> Visual) ---
                    # Process visual path in parallel with CPU operations
                    v_gw_latent_pre_fusion = global_workspace.gw_mod.encode({"v_latents": v_latent_vector})
                    
                    # Fuse in GW
                    presence_mask_visual = torch.ones(v_gw_latent_pre_fusion["v_latents"].size(0)).to(device)
                    gw_latent_from_v = global_workspace.gw_mod.fuse(
                        v_gw_latent_pre_fusion,
                        {"v_latents": presence_mask_visual}
                    )
                    
                    # Decode back to visual
                    half_cycle_v_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["v_latents"]
                    half_cycle_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(half_cycle_v_latent)[0]
                    half_cycle_image_np = half_cycle_image_tensor.permute(1, 2, 0).detach().cpu().numpy()
                    
                    # Get translated mask from future
                    translated_mask = translated_mask_future.result()
                    # Submit more CPU-bound tasks
                    translated_shape_color_future = executor.submit(extract_shape_color, translated_image_np, translated_mask)
                    half_cycle_mask_future = executor.submit(segment_shape, half_cycle_image_np)
                    
                    # --- Path 3: Full-Cycle (Visual -> GW -> Attribute -> GW -> Visual) ---
                    # Decode GW -> Attribute
                    intermediate_attr_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["attr"]
                    
                    # Encode back to GW
                    intermediate_attr_domain = global_workspace.encode_domain(intermediate_attr_latent, "attr")
                    intermediate_attr_gw_latent = global_workspace.gw_mod.encode({"attr": intermediate_attr_domain})
                    
                    # Fuse in GW
                    presence_mask_attr = torch.ones(intermediate_attr_gw_latent["attr"].size(0)).to(device)
                    gw_latent_intermediate = global_workspace.gw_mod.fuse(
                        intermediate_attr_gw_latent,
                        {"attr": presence_mask_attr}
                    )
                    
                    # Decode to visual
                    full_cycle_v_latent = global_workspace.gw_mod.decode(gw_latent_intermediate)["v_latents"]
                    full_cycle_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(full_cycle_v_latent)[0]
                    full_cycle_image_np = full_cycle_image_tensor.permute(1, 2, 0).cpu().numpy()
                    
                    # Get results from futures
                    half_cycle_mask = half_cycle_mask_future.result()
                    translated_shape_color = translated_shape_color_future.result() * 255
                    
                    # Submit remaining CPU tasks
                    half_cycle_shape_color_future = executor.submit(extract_shape_color, half_cycle_image_np, half_cycle_mask)
                    full_cycle_mask_future = executor.submit(segment_shape, full_cycle_image_np)
                    
                    # Get remaining results
                    half_cycle_shape_color = half_cycle_shape_color_future.result() * 255
                    full_cycle_mask = full_cycle_mask_future.result()
                    full_cycle_shape_color = extract_shape_color(full_cycle_image_np, full_cycle_mask) * 255
                
                # --- Store results ---
                processed_results.append({
                    'original_sample': sample,
                    'shape_color': full_cycle_shape_color,
                    'decoded_image': full_cycle_image_np,
                    'mask': full_cycle_mask,
                    # Translation Results
                    'translated_image': translated_image_np,
                    'translated_mask': translated_mask,
                    'translated_shape_color': translated_shape_color,
                    # Half-Cycle Results
                    'half_cycle_image': half_cycle_image_np,
                    'half_cycle_mask': half_cycle_mask,
                    'half_cycle_shape_color': half_cycle_shape_color,
                    # Full-Cycle Results
                    'full_cycle_image': full_cycle_image_np,
                    'full_cycle_mask': full_cycle_mask,
                    'full_cycle_shape_color': full_cycle_shape_color,
                })
                
        # Print progress
        print(f"Processed {min(i+batch_size, len(preprocessed_samples))}/{len(preprocessed_samples)} samples")

    return processed_results









# def process_through_global_workspace(
#     global_workspace: Any,
#     preprocessed_samples: List[Dict],
#     device: torch.device,
#     debug: bool = False,
# ) -> List[Dict]:
#     """
#     Processes samples through the Global Workspace, focusing on visual transformations.
#     Calculates:
#     1. Translation: Attribute -> GW -> Visual Image
#     2. Half-Cycle: Visual -> GW -> Visual Image
#     3. Full-Cycle: Visual -> GW -> Attribute -> GW -> Visual Image

#     Args:
#         global_workspace: The global workspace model instance.
#         preprocessed_samples: List of preprocessed samples. Each sample dict
#                               is expected to contain sample['model_inputs']
#                               with keys like 'attr_vector', 'one_hot', and 'v_latent'
#                               (initial visual latent representation).
#         device: Torch device.
#         debug: If True, generates random outputs instead of processing through model.

#     Returns:
#         List of processed samples, each containing the original sample info
#         and the decoded images (and their extracted colors) for the three paths.
#     """
#     processed_results = []
#     global_workspace.eval() # Make sure model is in evaluation mode

#     with torch.no_grad(): # Disable gradient calculations for inference
#         for sample in tqdm(preprocessed_samples, desc="Processing samples through GW model"):

#             if debug:
#                 import numpy as np
#                 # Debug mode - fast random output generation
#                 # Generate random outputs for debugging - KEEPING THIS FOR YOUR DEBUG NEEDS
#                 random_image = lambda: np.random.random((32, 32, 3))
#                 random_mask = lambda: np.random.random((32, 32)) > 0.5
#                 random_color = lambda: np.random.randint(0, 256, size=3)
#                 processed_results.append({
#                     'original_sample': sample,
#                     'translated_image': random_image(),
#                     'translated_mask': random_mask(),
#                     'translated_shape_color': random_color(),
#                     'half_cycle_image': random_image(),
#                     'half_cycle_mask': random_mask(),
#                     'half_cycle_shape_color': random_color(),
#                     'full_cycle_image': random_image(),
#                     'full_cycle_mask': random_mask(),
#                     'full_cycle_shape_color': random_color(),
#                 })
#                 continue

#             # --- Inputs ---
#             # Make sure inputs are correctly formatted and on the right device
#             # Attribute inputs
#             one_hot = sample['model_inputs']['one_hot'].to(device) # Assuming batch size 1, add unsqueeze(0) if needed
#             attr_vector = sample['model_inputs']['attr_vector'].to(device) # Assuming batch size 1, add unsqueeze(0) if needed
#             attr_inputs = [one_hot, attr_vector] # Or however your encode_domain expects them

#             # Visual inputs
#             # Ensure 'visual_ground_truth' key exists and contains the image data (e.g., numpy array HWC)
#             if 'visual_ground_truth' not in sample:
#                  raise ValueError(f"Sample missing 'visual_ground_truth'. Sample keys: {sample.keys()}")

#             img = sample["visual_ground_truth"]
#             # plot the image 
#             import matplotlib.pyplot as plt
            
#             # Convert PIL Image or Numpy array to Tensor CHW, add batch dim, send to device
#             visual_ground_truth_tensor = to_tensor(img)[:3].unsqueeze(0).to(device)
#             visual_module = global_workspace.domain_mods["v_latents"]
#             v_latent_vector = visual_module.visual_module.encode(visual_ground_truth_tensor)
#             # Get the initial domain-specific visual latent using the domain module's encoder
#             # Assuming encode_domain handles the visual_module.visual_module.encode() step internally
#             # v_latent_ground_truth_domain = global_workspace.encode_domain(visual_ground_truth_tensor, "v_latents") # Assuming this is the initial visual latent representation
            
#             # --- Path 1: Translation (Attribute -> GW -> Visual) ---
#             attr_gw_latent = global_workspace.gw_mod.encode(
#                 {"attr": global_workspace.encode_domain(attr_inputs, "attr")}
#             )
#             gw_latent_from_attr = global_workspace.gw_mod.fuse(
#                 attr_gw_latent,
#                 {"attr": torch.ones(attr_gw_latent["attr"].size(0)).to(device)}
#             )
#             # Decode GW -> Visual Latent
#             translated_v_latent = global_workspace.gw_mod.decode(gw_latent_from_attr)["v_latents"]
#             # Decode Visual Latent -> Image
#             # Assuming decode_images returns a list or a single tensor
#             translated_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(translated_v_latent)[0]
#             translated_image_np = translated_image_tensor.permute(1, 2, 0).detach().cpu().numpy()  # HWC format, numpy
#             translated_mask = segment_shape(translated_image_np)
#             translated_shape_color = extract_shape_color(translated_image_np, translated_mask) * 255

#             # --- Path 2: Half-Cycle (Visual -> GW -> Visual) ---
#             # Encode Initial Visual Latent -> GW
#             v_gw_latent_pre_fusion = global_workspace.gw_mod.encode({"v_latents": v_latent_vector})
            
#             # 2. Fuse the visual information in the GW
#             gw_latent_from_v = global_workspace.gw_mod.fuse(
#                 v_gw_latent_pre_fusion,
#                 {"v_latents": torch.ones(v_gw_latent_pre_fusion["v_latents"].size(0)).to(device)} # Presence mask
#             )


#             # 3. Decode the GW state back to the visual latent space
#             # Decode GW -> Visual Latent (Half Cycle complete)
#             half_cycle_v_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["v_latents"]
#             # Decode Visual Latent -> Image
#             half_cycle_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(half_cycle_v_latent)[0]
#             half_cycle_image_np = half_cycle_image_tensor.permute(1, 2, 0).detach().cpu().numpy()
#             half_cycle_mask = segment_shape(half_cycle_image_np)
#             half_cycle_shape_color = extract_shape_color(half_cycle_image_np, half_cycle_mask) * 255

#             # --- Path 3: Full-Cycle (Visual -> GW -> Attribute -> GW -> Visual) ---
#             # Start from gw_latent_from_v (Visual info in GW)
#             # Decode GW -> Attribute Latent
#             intermediate_attr_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["attr"]

#             # Encode Attribute Latent -> GW
#             # Again, assuming encode_domain handles latents or attr_vector format
#             intermediate_attr_domain = global_workspace.encode_domain(intermediate_attr_latent, "attr")
#             intermediate_attr_gw_latent = global_workspace.gw_mod.encode({"attr": intermediate_attr_domain})
#             gw_latent_intermediate = global_workspace.gw_mod.fuse(
#                 intermediate_attr_gw_latent,
#                 {"attr": torch.ones(intermediate_attr_gw_latent["attr"].size(0)).to(device)} # Presence mask
#             )
#             # Decode GW -> Visual Latent (Full Cycle complete)
#             full_cycle_v_latent = global_workspace.gw_mod.decode(gw_latent_intermediate)["v_latents"]
#             # Decode Visual Latent -> Image
#             full_cycle_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(full_cycle_v_latent)[0]
#             full_cycle_image_np = full_cycle_image_tensor.permute(1, 2, 0).cpu().numpy()
#             full_cycle_mask = segment_shape(full_cycle_image_np)
#             full_cycle_shape_color = extract_shape_color(full_cycle_image_np, full_cycle_mask) * 255

            
# ############################DEBUG IMAGES####################################################
#             debug_ = False
#             if debug_ == True :
#                 # plot the 3 images side to side 
#                 plt.figure()
#                 plt.axis('off')
#                 plt.title("Visual Ground Truth")
#                 # Convert to PIL Image or Numpy array if needed
#                 import numpy as np
#                 from PIL import Image
#                 if isinstance(img, np.ndarray):
#                     img = Image.fromarray(img)
#                 if isinstance(img, Image.Image):
#                     img = img.convert("RGB")
#                 # Add these lines before creating the PIL Image
#                 translated_image_np = np.clip(translated_image_np, 0, 1)  # Ensure values are in 0-1 range
#                 translated_image_np = (translated_image_np * 255).astype('uint8')
#                 half_cycle_image_np = np.clip(half_cycle_image_np, 0, 1)  # Ensure values are in 0-1 range
#                 half_cycle_image_np = (half_cycle_image_np * 255).astype('uint8')
#                 full_cycle_image_np = np.clip(full_cycle_image_np, 0, 1)  # Ensure values are in 0-1 range
#                 full_cycle_image_np = (full_cycle_image_np * 255).astype('uint8')

#                 if isinstance(translated_image_np, np.ndarray):
#                     img2 = Image.fromarray(translated_image_np)
#                     img3 = Image.fromarray(half_cycle_image_np)
#                     img4 = Image.fromarray(full_cycle_image_np)
#                 # plot the 4 on the same image
#                 plt.subplot(1, 4, 1)
#                 plt.imshow(img)
#                 plt.axis('off')
#                 plt.title("Visual Ground Truth")
#                 plt.subplot(1, 4, 2)
#                 plt.imshow(img2)
#                 plt.axis('off')
#                 plt.title("Translated Image")
#                 plt.subplot(1, 4, 3)
#                 plt.imshow(img3)
#                 plt.axis('off')
#                 plt.title("Half Cycle Image")
#                 plt.subplot(1, 4, 4)
#                 plt.imshow(img4)
#                 plt.axis('off')
#                 plt.title("Full Cycle Image")
#                 # Save the figure
#                 plt.savefig("image.png")
#                 plt.show()
#                 plt.close()
# ##############################################################################################""

#             # --- Store results ---
#             processed_results.append({
#                 'original_sample': sample, # Keep track of the original data
                
#                 # Translation Results
#                 'translated_image': translated_image_np,
#                 'translated_mask': translated_mask,
#                 'translated_shape_color': translated_shape_color, # RGB color
#                 # Half-Cycle Results
#                 'half_cycle_image': half_cycle_image_np,
#                 'half_cycle_mask': half_cycle_mask,
#                 'half_cycle_shape_color': half_cycle_shape_color, # RGB color
#                 # Full-Cycle Results
#                 'full_cycle_image': full_cycle_image_np,
#                 'full_cycle_mask': full_cycle_mask,
#                 'full_cycle_shape_color': full_cycle_shape_color, # RGB color
                
#             })

#     return processed_results











# def bin_processed_samples(
#     preprocessed_samples: Optional[List[dict]],
#     processed_samples: List[dict],
#     analysis_attributes: List[str],
#     binning_config: dict,
#     input_colors_by_attr: dict,
#     output_colors_by_attr: dict,
#     examples_by_attr: dict,
#     channels: List[str],
#     display_examples: bool
# ) -> None:
#     """
#     Bins processed samples according to the binning configuration.
    
#     Args:
#         preprocessed_samples: List of preprocessed samples (not used here, but kept for interface consistency).
#         processed_samples: List of processed samples.
#         analysis_attributes: List of attributes to analyze.
#         binning_config: Configuration for binning.
#         input_colors_by_attr: Dictionary to store input colors by attribute.
#         output_colors_by_attr: Dictionary to store output colors by attribute.
#         examples_by_attr: Dictionary to store examples by attribute.
#         channels: List of color channels.
#         display_examples: Whether to store example images.
#     """
#     for i, proc_result in enumerate(processed_samples):
#         orig_sample = proc_result['original_sample']
        
#         for attr in analysis_attributes:
#             attr_value = orig_sample[attr]
#             bin_idx = get_bin_index(attr_value, attr, binning_config)
#             bin_name = binning_config[attr]['bin_names'][bin_idx]
            
#             input_color = orig_sample['original_color']
#             for ch_idx, channel in enumerate(channels):
#                 input_colors_by_attr[attr][bin_name][channel].append(input_color[ch_idx])
            
#             output_color = proc_result['shape_color']
#             for ch_idx, channel in enumerate(channels):
#                 output_colors_by_attr[attr][bin_name][channel].append(output_color[ch_idx])
            
#             if display_examples and len(examples_by_attr[attr][bin_name]) < 10:
#                 vis_image = proc_result['decoded_image'].copy()
#                 mask = proc_result['mask']
                
#                 # Create a red border highlighting the detected shape
#                 kernel = np.ones((3, 3), np.uint8)
#                 mask_border = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) - mask.astype(np.uint8)
#                 contours_vis = np.zeros_like(vis_image)
#                 for c in range(3):
#                     contours_vis[:, :, c] = mask_border * (1.0 if c == 0 else 0.0)
#                 vis_image = np.clip(vis_image + contours_vis, 0, 1)
#                 examples_by_attr[attr][bin_name].append(vis_image)

# def bin_h_processed_samples(
#     preprocessed_samples,
#     processed_samples,
#     analysis_attributes,
#     binning_config,
#     input_colors_by_attr,
#     output_colors_by_attr,
#     examples_by_attr,
#     display_examples=True
# ):
#     """
#     Bin processed samples by attribute for H channel analysis.
#     """
#     for idx, (pre_sample, proc_sample) in enumerate(zip(preprocessed_samples, processed_samples)):
#         for attr in analysis_attributes:
#             # Instead of using bin_attribute(pre_sample, attr, binning_config),
#             # extract the attribute value and use get_bin_index.
#             attr_value = pre_sample[attr]
#             bin_idx = get_bin_index(attr_value, attr, binning_config)
#             bin_name = binning_config[attr]['bin_names'][bin_idx]
            
#             # Extract H channel from input RGB
#             input_h, _, _ = rgb_to_hsv(pre_sample['original_color']) # Use the first color for H value
#             # h_value = extract_h_channel(processed_sample, index)(pre_sample, idx)
#             if not np.isnan(input_h):
#                 input_colors_by_attr[attr][bin_name]['H'].append(input_h)
            
#             # Extract H channel from output RGB
#             output_h, _, _ = rgb_to_hsv(proc_sample["shape_color"])
#             if not np.isnan(output_h):
#                 output_colors_by_attr[attr][bin_name]['H'].append(output_h)
            
#             if display_examples and len(examples_by_attr[attr][bin_name]) < 10:
#                 vis_image = proc_sample['decoded_image'].copy()
#                 mask = proc_sample['mask']
                
#                 # Create a red border highlighting the detected shape
#                 kernel = np.ones((3, 3), np.uint8)
#                 mask_border = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) - mask.astype(np.uint8)
#                 contours_vis = np.zeros_like(vis_image)
#                 for c in range(3):
#                     contours_vis[:, :, c] = mask_border * (1.0 if c == 0 else 0.0)
#                 vis_image = np.clip(vis_image + contours_vis, 0, 1)
#                 examples_by_attr[attr][bin_name].append(vis_image)

# def get_bin_index(value: Any, attr: str, binning_config: dict) -> int:
#     """
#     Determines the bin index for a given attribute value based on the binning configuration.
    
#     Args:
#         value: The attribute value.
#         attr: The attribute name.
#         binning_config: The binning configuration.
        
#     Returns:
#         The bin index as an integer.
        
#     Raises:
#         ValueError: If the value cannot be binned (e.g., a categorical value not found in bin_names).
#     """
#     config = binning_config[attr]
    
#     # Handle categorical attributes (e.g., 'shape')
#     if attr == 'shape':
#         try:
#             return config['bin_names'].index(value)
#         except ValueError as e:
#             raise ValueError(f"Shape '{value}' not found in bin_names: {config['bin_names']}") from e
    
#     # For numerical attributes, compute or retrieve bin edges
#     if 'bin_edges' not in config:
#         n_bins = config['n_bins']
#         value_range = config['range']
#         bin_edges = np.linspace(value_range[0], value_range[1], n_bins + 1)
#     else:
#         bin_edges = config['bin_edges']
    
#     # Determine the bin index by scanning bin edges
#     for i in range(len(bin_edges) - 1):
#         if bin_edges[i] <= value < bin_edges[i + 1]:
#             return i
#     if value >= bin_edges[-1]:
#         return len(bin_edges) - 2
#     return 0