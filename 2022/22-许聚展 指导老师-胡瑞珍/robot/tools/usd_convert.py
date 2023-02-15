# --------------------------------------------------.
# obj to usd conversion.
# See : https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_asset-converter.html

# >>> https://github.com/ft-lab/omniverse_sample_scripts/blob/main/AssetConverter/importObjToUSD.py

# !!! run in script windows in isaac sim
# --------------------------------------------------.
import carb
import omni
import asyncio
# import omni.kit.asset_converter
import os

# Progress of processing.
def progress_callback (current_step: int, total: int):
    # Show progress
    print(f"{current_step} of {total}")

# Convert asset file(obj/fbx/glTF, etc) to usd.async 
async def convert_asset_to_usd (input_asset: str, output_usd: str):
    print("here")
    # Input options are defaults.
    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = False
    converter_context.ignore_camera = True
    converter_context.ignore_animations = True
    converter_context.ignore_light = True
    converter_context.export_preview_surface = False
    converter_context.use_meter_as_world_unit = False
    converter_context.create_world_as_default_root_prim = True
    converter_context.embed_textures = True
    converter_context.convert_fbx_to_y_up = False
    converter_context.convert_fbx_to_z_up = False
    converter_context.merge_all_meshes = False
    converter_context.use_double_precision_to_usd_transform_op = False 
    converter_context.ignore_pivots = False 
    converter_context.keep_all_materials = True
    converter_context.smooth_normals = True
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(input_asset, output_usd, progress_callback, converter_context)

    # Wait for completion.
    success = await task.wait_until_finished()
    if not success:
        print(input_asset)
        carb.log_error(task.get_status(), task.get_detailed_error())
    print("converting done")


def ycb():
    YCB_DIRECTORY = "E:/dataset/ycb"
    # usd_path = f'{YCB_DIRECTORY}/{obj_name}/google_16k/

    for model_folder in os.listdir(YCB_DIRECTORY):
    #for model_folder in ["007_tuna_fish_can"]:
        mf = os.path.join(YCB_DIRECTORY, model_folder, "google_16k")

        tf = os.path.join(YCB_DIRECTORY, model_folder, "google_16k", "textures")
        if os.path.exists(mf) and not os.path.exists(tf):
            print(model_folder)
            input_obj = os.path.join(mf, "textured.obj")
            output_usd = os.path.join(mf, "text.usd")
            # convert_asset_to_usd(input_obj, output_usd)
            
            asyncio.ensure_future(
                convert_asset_to_usd(input_obj, output_usd))


GN_DIRECTORY = "E:/dataset/grasp_net/models"

for model_folder in os.listdir(GN_DIRECTORY):
#for model_folder in ["022"]:
    mf = os.path.join(GN_DIRECTORY, model_folder)
    if not os.path.isdir(mf):
        continue
    tf = os.path.join(mf, "omni", "textures")
    if os.path.exists(mf) and not os.path.exists(tf):
        print(model_folder)
        input_obj = os.path.join(mf, "simple.dae")
        output_usd = os.path.join(mf, "omni", "simple.usd")
        # convert_asset_to_usd(input_obj, output_usd)
        
        asyncio.ensure_future(
            convert_asset_to_usd(input_obj, output_usd))
