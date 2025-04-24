# ----- Helper Functions (Add to SSD_utils.py or keep here) -----
import io
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import colorsys # Needed for rgb_to_hsv implementation provided

# Import generate_image from its location
from simple_shapes_dataset.cli import generate_image

# Make sure LOGGER is defined if used (e.g., import logging; LOGGER = logging.getLogger(__name__))
import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Example basic config

# Provided rgb_to_hsv function
def rgb_to_hsv(rgb_values):
    """
    Convert RGB values (0-255) to HSV values
    Returns H in range [0,360), S and V in range [0,1]
    """
    if rgb_values is None:
        return np.nan, np.nan, np.nan
    # Normalize RGB to [0,1] range
    r, g, b = [x/255.0 for x in rgb_values]
    # Use try-except for potential errors in colorsys with pure black/white/grey
    try:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
    except Exception: # Catch potential division by zero etc.
        h, s, v = 0, 0, v # Assign H=0, S=0 if conversion fails but V is valid
    # Convert H to degrees [0,360)
    h = h * 360
    return h, s, v

# Image Generation Helper
def generate_image_tensor(
    cls: int,
    pos: np.ndarray,
    size: int,
    rotation: float,
    color_rgb_0_255: Union[np.ndarray, Tuple[int, int, int]], # Expect RGB 0-255
    imsize: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generates an image using matplotlib based on attributes and returns it as a PyTorch tensor.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(imsize/dpi, imsize/dpi), dpi=dpi, frameon=False)
    ax.set_axis_off()
    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)
    ax.set_aspect('equal', adjustable='box')

    # generate_image expects color as 0-255 numpy array
    color_np = np.array(color_rgb_0_255)

    # --- Call the existing generate_image function ---
    generate_image(ax, cls, pos, size, rotation, color_np, imsize)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    pil_img = Image.open(buf).convert('RGB')
    pil_img = pil_img.resize((imsize, imsize), Image.Resampling.LANCZOS)
    img_tensor = TF.to_tensor(pil_img) # Converts to (C, H, W) and scales to [0, 1]
    return img_tensor.to(device)

# Modified Preprocessing Function
def preprocess_dataset(
    df: pd.DataFrame,
    attributes_for_vector: List[str], # Attributs à inclure dans attr_vector
    shape_names: List[str],
    color_attr_in_model: bool, # True si le modèle attend la couleur dans attr_vector
    rgb_colors_list: np.ndarray, # Liste complète des couleurs RGB 0-255
    device: torch.device,
    visual_encoder: torch.nn.Module, # L'encodeur VAE (e.g., vae.encoder)
    imsize: int = 32,
) -> List[Dict]:
    """
    Preprocesses samples from a DataFrame by GENERATING images on the fly.
    Encodes images to get initial v_latent (mean of VAE).
    Prepares attribute vectors based on whether color is included.
    """
    preprocessed_samples = []
    required_cols = ['class', 'location', 'size', 'rotation', 'color_index']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame manque des colonnes nécessaires. Requis: {required_cols}, Trouvé: {df.columns.tolist()}")

    if isinstance(df["location"].iloc[0], str):
         df["location"] = df["location"].apply(ast.literal_eval)

    visual_encoder.eval() # Ensure encoder is in eval mode

    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing (generating images)"):
            cls = int(row['class'])
            loc = np.array(row['location'])
            size = int(row['size'])
            rotation = float(row['rotation'])
            color_idx = int(row['color_index'])
            rgb_color_0_255 = rgb_colors_list[color_idx]

            image_tensor = generate_image_tensor(
                cls, loc, size, rotation, rgb_color_0_255, imsize, device
            ) # (C, H, W), [0, 1]

            # Encode image: encode returns (mu, logvar)
            mean, logvar = visual_encoder.encode(image_tensor.unsqueeze(0))
            initial_v_latent = mean # Use mean as the latent representation

            one_hot = torch.zeros(len(shape_names), device=device)
            one_hot[cls] = 1.0

            norm_size = normalize_size(size, min_val=7, max_val=14)
            norm_rot = normalize_rotation(rotation)
            norm_pos_x = normalize_position(loc[0], max_val=imsize)
            norm_pos_y = normalize_position(loc[1], max_val=imsize)

            attr_list = []
            if 'size' in attributes_for_vector: attr_list.append(norm_size)
            if 'rotation' in attributes_for_vector: attr_list.append(norm_rot)
            if 'position_x' in attributes_for_vector: attr_list.append(norm_pos_x)
            if 'position_y' in attributes_for_vector: attr_list.append(norm_pos_y)

            if color_attr_in_model:
                h, s, v = rgb_to_hsv(rgb_color_0_255) # H(0-360), S(0-1), V(0-1)
                norm_h = h / 360.0 if not np.isnan(h) else 0.0
                norm_s = s if not np.isnan(s) else 0.0
                norm_v = v if not np.isnan(v) else 0.0
                # Assuming order is H, S, V if color is included
                if 'color' in attributes_for_vector: # Check if 'color' group is expected
                     attr_list.extend([norm_h, norm_s, norm_v])

            attr_vector = torch.tensor(attr_list, dtype=torch.float32, device=device)

            sample_data = {
                'index': index,
                'ground_truth': {
                    'class': cls, 'location': loc, 'size': size, 'rotation': rotation,
                    'color_index': color_idx, 'rgb_color': tuple(rgb_color_0_255),
                    'image_tensor': image_tensor # Store the generated tensor
                },
                'model_inputs': {
                    'one_hot': one_hot.unsqueeze(0),
                    'attr_vector': attr_vector.unsqueeze(0),
                    'v_latent': initial_v_latent, # Already has batch dim
                }
            }
            preprocessed_samples.append(sample_data)

    return preprocessed_samples

# Rewritten Processing Function
def process_through_global_workspace(
    global_workspace: Any,
    preprocessed_samples: List[Dict],
    device: torch.device,
    debug: bool = False,
) -> List[Dict]:
    """
    Processes samples through the Global Workspace for three paths:
    1. Translation: Attribute -> GW -> Visual
    2. Half-Cycle: Visual -> GW -> Visual
    3. Full-Cycle: Visual -> GW -> Attribute -> GW -> Visual
    Decodes images and extracts colors for each path.
    """
    processed_results = []
    global_workspace.eval() # Ensure model is in evaluation mode

    # --- Get necessary model components ---
    try:
        visual_domain = global_workspace.domain_mods["v_latents"]
        attr_domain = global_workspace.domain_mods["attr"] # Assuming this exists
        gw_mod = global_workspace.gw_mod
    except KeyError as e:
        raise KeyError(f"Clé manquante dans global_workspace.domain_mods: {e}. "
                       f"Clés disponibles: {list(global_workspace.domain_mods.keys())}") from e
    except AttributeError as e:
        raise AttributeError(f"Attribut manquant dans global_workspace: {e}") from e

    with torch.no_grad():
        for sample in tqdm(preprocessed_samples, desc="Processing through GW"):

            if debug: # Keep debug block if useful
                # ... (debug block code as before) ...
                continue

            # --- Inputs ---
            if 'model_inputs' not in sample:
                 LOGGER.warning(f"Skipping sample (index {sample.get('index', 'N/A')}) due to missing 'model_inputs'.")
                 continue
            model_inputs = sample['model_inputs']

            try:
                one_hot = model_inputs['one_hot'].to(device)
                attr_vector = model_inputs['attr_vector'].to(device)
                # Check if encode_domain expects list or tensor
                # Assuming it expects the raw tensor from the vector list
                attr_inputs = attr_vector # Or adapt if it needs one_hot too
                initial_v_latent = model_inputs['v_latent'].to(device)
                batch_size = initial_v_latent.size(0) # Should be 1 here
                if batch_size != 1:
                    LOGGER.warning(f"Processing sample (index {sample.get('index', 'N/A')}) with batch size {batch_size} might be unexpected.")

            except KeyError as e:
                LOGGER.warning(f"Skipping sample (index {sample.get('index', 'N/A')}) due to missing key in 'model_inputs': {e}")
                continue
            except Exception as e:
                 LOGGER.error(f"Error processing inputs for sample (index {sample.get('index', 'N/A')}): {e}")
                 continue

            result_dict = {'original_sample': sample}

            # --- Helper to decode and extract color ---
            def decode_extract(v_latent_tensor):
                if v_latent_tensor is None: return None, None, None
                try:
                    img_tensor = visual_domain.decode_images(v_latent_tensor)
                    if isinstance(img_tensor, list): img_tensor = img_tensor[0]
                    if img_tensor.ndim == 4: img_tensor = img_tensor.squeeze(0)
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    # Clip values just in case they are slightly outside [0, 1]
                    img_np = np.clip(img_np, 0.0, 1.0)
                    mask = segment_shape(img_np) # Assumes segment_shape handles 0-1 range
                    # Assumes extract_shape_color returns normalized [0,1] RGB
                    color_norm = extract_shape_color(img_np, mask)
                    if color_norm is None or np.isnan(color_norm).any():
                        return img_np, None, None # Return image even if color extraction fails
                    color_0_255 = (color_norm * 255).astype(np.uint8)
                    return img_np, mask, color_0_255
                except Exception as e:
                    LOGGER.error(f"Error during decode/extract for sample {sample.get('index', 'N/A')}: {e}")
                    return None, None, None # Return None on error

            # --- Path 1: Translation (Attribute -> GW -> Visual) ---
            try:
                # Note: Adjust encode_domain call based on its actual signature
                attr_latent_domain = attr_domain.encode(attr_inputs) # Assuming domain module handles encoding
                attr_gw_latent = gw_mod.encode({"attr": attr_latent_domain})
                presence_attr = {"attr": torch.ones(batch_size).to(device)}
                gw_latent_from_attr = gw_mod.fuse(attr_gw_latent, presence_attr)
                translated_v_latent = gw_mod.decode(gw_latent_from_attr).get("v_latents") # Use .get for safety
                img, mask, color = decode_extract(translated_v_latent)
                result_dict['translated_image'] = img
                result_dict['translated_mask'] = mask
                result_dict['translated_shape_color'] = color # RGB 0-255 or None
            except Exception as e:
                LOGGER.error(f"Error in Translation path for sample {sample.get('index', 'N/A')}: {e}")
                result_dict.update({'translated_image': None, 'translated_mask': None, 'translated_shape_color': None})


            # --- Path 2: Half-Cycle (Visual -> GW -> Visual) ---
            try:
                # Assuming visual_domain.encode just passes latent through if input is already latent
                v_latent_domain_initial = visual_domain.encode(initial_v_latent)
                v_gw_latent_initial = gw_mod.encode({"v_latents": v_latent_domain_initial})
                presence_v = {"v_latents": torch.ones(batch_size).to(device)}
                gw_latent_from_v = gw_mod.fuse(v_gw_latent_initial, presence_v)
                half_cycle_v_latent = gw_mod.decode(gw_latent_from_v).get("v_latents")
                img, mask, color = decode_extract(half_cycle_v_latent)
                result_dict['half_cycle_image'] = img
                result_dict['half_cycle_mask'] = mask
                result_dict['half_cycle_shape_color'] = color # RGB 0-255 or None
            except Exception as e:
                LOGGER.error(f"Error in Half-Cycle path for sample {sample.get('index', 'N/A')}: {e}")
                result_dict.update({'half_cycle_image': None, 'half_cycle_mask': None, 'half_cycle_shape_color': None})


            # --- Path 3: Full-Cycle (Visual -> GW -> Attribute -> GW -> Visual) ---
            try:
                # Need gw_latent_from_v from Path 2
                if 'gw_latent_from_v' in locals(): # Check if Path 2 succeeded
                    intermediate_attr_latent = gw_mod.decode(gw_latent_from_v).get("attr")
                    if intermediate_attr_latent is not None:
                        intermediate_attr_domain = attr_domain.encode(intermediate_attr_latent)
                        intermediate_attr_gw_latent = gw_mod.encode({"attr": intermediate_attr_domain})
                        presence_attr_intermediate = {"attr": torch.ones(intermediate_attr_latent.size(0)).to(device)}
                        gw_latent_intermediate = gw_mod.fuse(intermediate_attr_gw_latent, presence_attr_intermediate)
                        full_cycle_v_latent = gw_mod.decode(gw_latent_intermediate).get("v_latents")
                        img, mask, color = decode_extract(full_cycle_v_latent)
                        result_dict['full_cycle_image'] = img
                        result_dict['full_cycle_mask'] = mask
                        result_dict['full_cycle_shape_color'] = color # RGB 0-255 or None
                    else: raise ValueError("Intermediate attribute latent was None.")
                else: raise ValueError("Half-cycle path failed, cannot perform full-cycle.")
            except Exception as e:
                LOGGER.error(f"Error in Full-Cycle path for sample {sample.get('index', 'N/A')}: {e}")
                result_dict.update({'full_cycle_image': None, 'full_cycle_mask': None, 'full_cycle_shape_color': None})

            processed_results.append(result_dict)

    return processed_results


# Modified Binning Initialization
def initialize_h_binning_structures(
    analysis_attributes,
    binning_config
):
    """
    Initialize data structures for binning samples by attribute for H channel.
    Stores Ground Truth H and H from the three processing paths.
    Also prepares lists for example images.
    """
    input_colors_by_attr = {}
    output_colors_by_attr = {}
    examples_by_attr = {} # Stores tuples of (gt_img, trans_img, half_img, full_img)

    for attr in analysis_attributes:
        attr_bins = {}
        output_bins = {}
        example_bins = {}
        for bin_name in binning_config[attr]['bin_names']:
            # Structure to hold lists of Hue values for each path
            attr_bins[bin_name] = {
                'H_gt': [],
                'H_translated': [],
                'H_half_cycle': [],
                'H_full_cycle': [],
            }
            # Separate structure or integrate? Let's integrate for simplicity
            # output_bins[bin_name] = {} # No longer needed if integrated
            example_bins[bin_name] = [] # List to store example tuples/dicts

        input_colors_by_attr[attr] = attr_bins
        # output_colors_by_attr[attr] = output_bins # No longer needed
        examples_by_attr[attr] = example_bins

    return colors_by_attr, examples_by_attr # Return only 2 dicts now

# Modified Binning Function
def bin_h_processed_samples(
    preprocessed_samples: List[Dict], # Contains ground truth info
    processed_samples: List[Dict],    # Contains processed results
    analysis_attributes: List[str],
    binning_config: Dict,
    colors_by_attr: Dict,             # Renamed from input_colors_by_attr
    examples_by_attr: Dict,
    max_examples_per_bin: int = 5,
    display_examples: bool = True # Kept for consistency, but examples stored differently
):
    """
    Bins samples based on ground truth attributes and collects Hue values
    (ground truth, translated, half-cycle, full-cycle) and example images.
    """
    if len(preprocessed_samples) != len(processed_samples):
        LOGGER.error("Mismatch between preprocessed and processed sample counts!")
        return # Or raise error

    for i, pre_sample in enumerate(preprocessed_samples):
        proc_sample = processed_samples[i]
        gt = pre_sample['ground_truth']

        for attr in analysis_attributes:
            if attr not in gt: continue # Skip if attribute not in ground truth keys

            try:
                 # Use the bin_attribute function (ensure it exists in SSD_utils)
                 # It needs the value, the config for that attribute
                 value_to_bin = gt[attr]
                 # Handle specific cases like shape class index -> name
                 if attr == 'shape':
                      # Assuming shape_names is accessible or passed somehow,
                      # or bin_attribute handles index directly if config has no names
                      # Let's assume bin_attribute can handle the class index if bin_names exist
                      pass # Keep value_to_bin as class index if bin_attribute handles it
                 elif attr == 'location':
                      # Binning location might need splitting into x/y first
                      # This part depends heavily on how bin_attribute works for location
                      # Skipping location binning for now unless bin_attribute handles it
                      continue

                 bin_name = bin_attribute(value_to_bin, attr, binning_config)

                 if bin_name is not None and bin_name in colors_by_attr.get(attr, {}):
                     target_bin = colors_by_attr[attr][bin_name]

                     # --- Collect Hue Values ---
                     # Ground Truth
                     h_gt, _, _ = rgb_to_hsv(gt['rgb_color'])
                     if not np.isnan(h_gt): target_bin['H_gt'].append(h_gt)

                     # Translated
                     rgb_trans = proc_sample.get('translated_shape_color')
                     if rgb_trans is not None:
                         h_trans, _, _ = rgb_to_hsv(rgb_trans)
                         if not np.isnan(h_trans): target_bin['H_translated'].append(h_trans)

                     # Half-Cycle
                     rgb_half = proc_sample.get('half_cycle_shape_color')
                     if rgb_half is not None:
                         h_half, _, _ = rgb_to_hsv(rgb_half)
                         if not np.isnan(h_half): target_bin['H_half_cycle'].append(h_half)

                     # Full-Cycle
                     rgb_full = proc_sample.get('full_cycle_shape_color')
                     if rgb_full is not None:
                         h_full, _, _ = rgb_to_hsv(rgb_full)
                         if not np.isnan(h_full): target_bin['H_full_cycle'].append(h_full)

                     # --- Collect Examples ---
                     if len(examples_by_attr[attr][bin_name]) < max_examples_per_bin:
                         example_data = {
                             'gt_img': pre_sample['ground_truth']['image_tensor'].cpu(), # Store tensor
                             'gt_color': gt['rgb_color'],
                             'trans_img': proc_sample.get('translated_image'), # Store np array or tensor
                             'trans_color': proc_sample.get('translated_shape_color'),
                             'half_img': proc_sample.get('half_cycle_image'),
                             'half_color': proc_sample.get('half_cycle_shape_color'),
                             'full_img': proc_sample.get('full_cycle_image'),
                             'full_color': proc_sample.get('full_cycle_shape_color'),
                         }
                         examples_by_attr[attr][bin_name].append(example_data)

            except Exception as e:
                LOGGER.error(f"Error binning sample {i} for attribute '{attr}': {e}")
                # Continue to next attribute or sample

# ----- Main Class -----

class HueShapeAnalyzer:
    """
    Analyzes shape data by generating images on the fly, processing through GW paths,
    and comparing Hue distributions.
    """
    def __init__(
        self,
        global_workspace: GlobalWorkspace2Domains, # Expect the loaded model
        device: torch.device,
        shape_names: List[str] = ["diamond", "egg", "triangle"],
        color_attr_in_model: bool = True, # True if model uses color attribute
        output_dir: str = ".",
        seed: int = 0,
        imsize: int = 32,
    ):
        """Initialize the analyzer."""
        self.global_workspace = global_workspace.to(device).eval() # Move to device and set eval
        self.device = device
        self.shape_names = shape_names
        self.color_attr_in_model = color_attr_in_model # Store flag
        self.output_dir = output_dir
        self.seed = seed
        self.imsize = imsize # Store image size

        os.makedirs(self.output_dir, exist_ok=True)
        # Generate reference colors (ensure generate_fixed_colors is available)
        self.rgb_colors, _ = generate_fixed_colors(100) # Assuming 100 colors
        self.binning_config = default_binning_config # Use default or allow override

        # Get and store the visual encoder upon initialization
        self.visual_encoder = self._get_visual_encoder()


    def _get_visual_encoder(self) -> torch.nn.Module:
        """Retrieves the visual VAE encoder module from the global workspace."""
        try:
            # Access based on Point 2: VisualLatentDomainModule -> visual_module -> vae -> encoder
            visual_domain_container = self.global_workspace.domain_mods["v_latents"]
            if isinstance(visual_domain_container, VisualLatentDomainModule) or \
               isinstance(visual_domain_container, VisualLatentDomainWithUnpairedModule): # Handle both types
                visual_module = visual_domain_container.visual_module
                if hasattr(visual_module, 'vae') and isinstance(visual_module.vae, VAE):
                    visual_encoder = visual_module.vae.encoder
                else:
                    raise AttributeError("Structure mismatch: visual_module has no 'vae' or 'vae' is not a VAE instance.")
            elif isinstance(visual_domain_container, VisualDomainModule): # Direct case (less likely based on config)
                 if hasattr(visual_domain_container, 'vae') and isinstance(visual_domain_container.vae, VAE):
                     visual_encoder = visual_domain_container.vae.encoder
                 else:
                    raise AttributeError("Structure mismatch: VisualDomainModule has no 'vae' or 'vae' is not a VAE instance.")
            else:
                raise TypeError(f"Unexpected type for domain_mods['v_latents']: {type(visual_domain_container)}")

        except KeyError:
             raise KeyError("'v_latents' not found in global_workspace.domain_mods")
        except AttributeError as e:
             raise AttributeError(f"Error accessing encoder structure: {e}")

        if visual_encoder is None:
             raise ValueError("visual_encoder could not be extracted.")

        return visual_encoder.eval() # Return encoder in eval mode

    def _load_and_process_data( # Renamed from _process_csv for clarity
        self,
        csv_path: str,
        attributes_for_vector: List[str], # Attributes expected by the model's attr vector
    ) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
        """
        Loads CSV, preprocesses samples (generating images + initial v_latent),
        and processes them through the GW paths.
        """
        df = pd.read_csv(csv_path)

        LOGGER.info("Starting preprocessing (image generation and encoding)...")
        preprocessed_samples = preprocess_dataset(
            df=df,
            attributes_for_vector=attributes_for_vector,
            shape_names=self.shape_names,
            color_attr_in_model=self.color_attr_in_model,
            rgb_colors_list=self.rgb_colors,
            device=self.device,
            visual_encoder=self.visual_encoder, # Use stored encoder
            imsize=self.imsize,
        )
        LOGGER.info("Preprocessing finished.")

        LOGGER.info("Starting processing through Global Workspace paths...")
        processed_samples = process_through_global_workspace(
            global_workspace=self.global_workspace,
            preprocessed_samples=preprocessed_samples,
            device=self.device,
        )
        LOGGER.info("Processing finished.")

        return df, preprocessed_samples, processed_samples

    def analyze_dataset(
        self,
        csv_path: str,
        analysis_attributes: List[str] = None, # Attributes to BIN BY for analysis
        model_attributes: List[str] = None,   # Attributes model uses in its vector
        display_examples: bool = True,
        seed: Optional[int] = None,
        binning_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyzes dataset by generating images, processing through GW, binning results,
        and performing initial analysis (e.g., comparing input vs output for one path).
        """
        if analysis_attributes is None:
            analysis_attributes = ['shape', 'rotation', 'size', 'position_x', 'position_y']
        if model_attributes is None: # Default attributes model might use
            model_attributes = ['size', 'rotation', 'position_x', 'position_y']
            if self.color_attr_in_model:
                 model_attributes.append('color') # Group name for H, S, V
        current_binning_config = binning_config if binning_config is not None else self.binning_config
        current_seed = seed if seed is not None else self.seed
        seed_everything(current_seed) # Set seed if provided

        results = {attr: {} for attr in analysis_attributes} # Results structure

        # --- Load, preprocess, and process data ---
        df, preprocessed_samples, processed_samples = self._load_and_process_data(
            csv_path=csv_path,
            attributes_for_vector=model_attributes
        )

        # --- Initialize binning structures ---
        # Now stores H for GT, Trans, Half, Full
        colors_by_attr, examples_by_attr = initialize_h_binning_structures(
            analysis_attributes,
            current_binning_config
        )

        # --- Bin the results ---
        LOGGER.info("Binning processed samples...")
        bin_h_processed_samples(
            preprocessed_samples=preprocessed_samples,
            processed_samples=processed_samples,
            analysis_attributes=analysis_attributes,
            binning_config=current_binning_config,
            colors_by_attr=colors_by_attr,
            examples_by_attr=examples_by_attr,
            display_examples=display_examples # Flag now mainly controls example collection count
        )
        LOGGER.info("Binning finished.")

        # --- Perform Analysis and Visualization (NEEDS ADAPTATION) ---
        # The existing `process_analysis_attributes` compares input vs *one* output.
        # We now have GT vs THREE outputs. We need to decide what to compare/visualize here.

        # Option 1: Compare GT vs Full-Cycle (most relevant for synesthesia?)
        LOGGER.info("Processing analysis attributes (comparing GT vs Full-Cycle Hue)...")
        # We need a modified compare_bin_in_out and process_analysis_attributes
        # OR adapt the data structure passed to the existing ones.

        # Let's simulate adapting the data structure for the *existing* functions temporarily:
        # Create temporary dicts mimicking the old input/output structure for ONE path
        temp_input_colors = {}
        temp_output_colors = {} # Using Full-Cycle output
        for attr in analysis_attributes:
            temp_input_colors[attr] = {}
            temp_output_colors[attr] = {}
            for bin_name in current_binning_config[attr]['bin_names']:
                 if bin_name in colors_by_attr.get(attr, {}):
                     temp_input_colors[attr][bin_name] = {'H': colors_by_attr[attr][bin_name]['H_gt']}
                     temp_output_colors[attr][bin_name] = {'H': colors_by_attr[attr][bin_name]['H_full_cycle']}

        # Call the *existing* process_analysis_attributes with the adapted data
        # NOTE: This only analyzes ONE output path (Full-Cycle here)
        results_processed = process_analysis_attributes(
            analysis_attributes=analysis_attributes,
            output_dir=self.output_dir,
            color=self.color_attr_in_model, # Flag for naming output folder correctly
            binning_config=current_binning_config,
            input_colors_by_attr=temp_input_colors, # Adapted GT data
            output_colors_by_attr=temp_output_colors, # Adapted Full-Cycle data
            examples_by_attr=examples_by_attr, # Pass examples (vis func needs adapting too)
            results={}, # Start with fresh results dict for this part
        )
        # Store these specific results
        results['gt_vs_full_cycle_analysis'] = results_processed

        # Store the full binned data for potential later use
        results['binned_hue_data'] = colors_by_attr
        results['binned_examples'] = examples_by_attr

        LOGGER.info("Dataset analysis finished.")
        return results # Return comprehensive results

    def compare_hue_distributions_across_shapes(
        self,
        csv_path: str,
        model_attributes: List[str] = None, # Attributes model uses in vector
        shape_names: Optional[List[str]] = None,
        comparison_path: Literal['translated', 'half_cycle', 'full_cycle'] = 'full_cycle', # Choose path to compare
        display_distributions: bool = True,
        display_ks_test: bool = True,
        display_kl_divergence: bool = True,
    ) -> Dict[str, Any]:
        """
        Compares reconstructed Hue distributions (from a chosen path) across shapes.
        Generates images on the fly.
        """
        output_subdir = f"shape_hue_comparison_{comparison_path}"
        output_path_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_path_dir, exist_ok=True)

        if shape_names is None:
            shape_names = self.shape_names
        else: # Validate provided shape names
            if not all(sn in self.shape_names for sn in shape_names):
                 raise ValueError(f"Provided shape_names {shape_names} contain names not in analyzer's list: {self.shape_names}")

        if model_attributes is None: # Default attributes model might use
            model_attributes = ['size', 'rotation', 'position_x', 'position_y']
            if self.color_attr_in_model:
                 model_attributes.append('color')

        LOGGER.info(f"Starting shape comparison using '{comparison_path}' path...")
        # --- Load, preprocess, and process data ---
        # Only need 'shape' attribute for model vector if compare_hue doesn't depend on others
        # But preprocess_dataset needs all attributes to generate image, so pass model_attributes
        df, _, processed_samples = self._load_and_process_data(
            csv_path=csv_path,
            attributes_for_vector=model_attributes # Pass attributes needed by model
        )

        # --- Extract Hue for the chosen path for each shape ---
        shape_hue_distributions = {shape_name: {'H': []} for shape_name in shape_names}
        output_color_key = f"{comparison_path}_shape_color" # e.g., 'full_cycle_shape_color'

        if 'class' not in df.columns:
            raise ValueError("CSV file must contain a 'class' column for shape index.")

        for index, row in df.iterrows():
            shape_index = int(row['class'])
            if 0 <= shape_index < len(self.shape_names):
                shape_label = self.shape_names[shape_index]
            else: continue # Skip invalid class index

            if shape_label in shape_names:
                if index < len(processed_samples):
                    proc_sample = processed_samples[index]
                    output_color_rgb = proc_sample.get(output_color_key)

                    if output_color_rgb is not None:
                        h_value, _, _ = rgb_to_hsv(output_color_rgb)
                        if not np.isnan(h_value):
                            shape_hue_distributions[shape_label]['H'].append(h_value)
                else:
                     LOGGER.warning(f"Index {index} out of bounds for processed_samples.")

        # --- Perform Comparisons and Visualizations ---
        num_bins = 50
        if display_distributions:
            # Visualize distributions for the chosen path
            visualize_color_distributions_by_attribute(
                color_data=shape_hue_distributions,
                attribute_name="shape",
                bin_names=shape_names,
                output_path=os.path.join(output_path_dir, f'shape_hue_dist_{comparison_path}.png'),
                channels=["H"],
                num_bins=num_bins
            )

        ks_test_results = {}
        kl_divergence_results = {}
        shape_pairs = []
        for i in range(len(shape_names)):
            for j in range(i + 1, len(shape_names)):
                shape_pairs.append((shape_names[i], shape_names[j]))

        for shape1, shape2 in shape_pairs:
            ks_test_results[(shape1, shape2)] = {}
            kl_divergence_results[(shape1, shape2)] = {}
            dist1 = np.array(shape_hue_distributions[shape1]['H'])
            dist2 = np.array(shape_hue_distributions[shape2]['H'])

            if len(dist1) > 1 and len(dist2) > 1:
                # KS Test
                ks_stat, ks_pval = ks_2samp(dist1, dist2)
                ks_test_results[(shape1, shape2)]['H'] = {'ks_statistic': ks_stat, 'p_value': ks_pval}
                print(f"KS test ({comparison_path}) {shape1} vs {shape2}, Hue: stat={ks_stat:.3f}, p={ks_pval:.3f}")

                # KL Divergence
                kl_1_to_2 = kl_divergence(dist1, dist2) # Assumes kl_divergence handles samples
                kl_2_to_1 = kl_divergence(dist2, dist1)
                symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2 if kl_1_to_2 is not None and kl_2_to_1 is not None else None
                kl_divergence_results[(shape1, shape2)]['H'] = {
                    'kl_1_to_2': kl_1_to_2, 'kl_2_to_1': kl_2_to_1, 'kl_symmetric': symmetric_kl
                }
            else:
                # Handle insufficient data
                ks_test_results[(shape1, shape2)]['H'] = {'error': 'Insufficient data'}
                kl_divergence_results[(shape1, shape2)]['H'] = {'error': 'Insufficient data'}
                LOGGER.warning(f"Insufficient data for {shape1} vs {shape2} ({comparison_path})")


        # --- Generate Heatmaps and Summary (Adapted for specific path) ---
        # (Heatmap/Summary generation code largely unchanged, just update titles/filenames)
        summary_path = os.path.join(output_path_dir, f'shape_hue_comparison_summary_{comparison_path}.txt')
        # ... [Code to generate heatmaps and summary file, ensuring titles/filenames include comparison_path] ...
        # (Using the existing heatmap/summary code from your snippet, just ensure filenames differ)
        # Example for KS heatmap filename:
        # plt.savefig(os.path.join(output_path_dir, f'shape_hue_ks_test_heatmap_{comparison_path}.png'))

        LOGGER.info(f"Shape comparison for path '{comparison_path}' finished.")
        return {
            'comparison_path': comparison_path,
            'ks_test_results': ks_test_results,
            'kl_divergence_results': kl_divergence_results
        }