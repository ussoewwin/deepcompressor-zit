"""
Vast.ai用事前検証スクリプト
量子化実行前にZImageTransformerStructが正しく構築されるか確認
"""
import sys
import os

# パス設定（Vast.ai環境用）
sys.path.insert(0, "/root/deepcompressor-zit")
os.chdir("/root/deepcompressor-zit")

def main():
    print("=" * 80)
    print("ZIT構造体事前検証")
    print("=" * 80)
    
    # 1. インポート確認
    print("\n=== 1. インポート確認 ===")
    try:
        from deepcompressor.app.diffusion.nn.struct import (
            DiffusionModelStruct,
            DiTStruct,
            ZImageTransformerStruct,
        )
        print("  DiffusionModelStruct: OK")
        print("  DiTStruct: OK")
        print("  ZImageTransformerStruct: OK")
    except ImportError as e:
        print(f"  インポートエラー: {e}")
        return False
    
    # 2. ZITモデルロードと構造体構築
    print("\n=== 2. ZITモデル構造体構築 ===")
    try:
        import torch
        from diffusers import ZImagePipeline
        
        print("  ZImagePipelineをロード中...")
        pipeline = ZImagePipeline.from_pretrained(
            "Zluda/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
        )
        transformer = pipeline.transformer
        print(f"  transformerの型: {type(transformer)}")
        print(f"  transformer.layers数: {len(transformer.layers)}")
        print(f"  transformer.context_refiner数: {len(transformer.context_refiner)}")
        print(f"  transformer.noise_refiner数: {len(transformer.noise_refiner)}")
        
        # 構造体を構築
        print("\n  DiffusionModelStruct.construct() を呼び出し中...")
        model_struct = DiffusionModelStruct.construct(pipeline)
        
        # 型確認
        struct_type = type(model_struct).__name__
        print(f"  構築された構造体の型: {struct_type}")
        
        if struct_type == "ZImageTransformerStruct":
            print("  SUCCESS: ZImageTransformerStructとして構築されました")
        else:
            print(f"  FAILURE: {struct_type}として構築されました")
            return False
        
        # 3. block_structs確認
        print("\n=== 3. block_structs確認 ===")
        block_structs = model_struct.block_structs
        print(f"  block_structs数: {len(block_structs)}")
        
        refiner_count = 0
        layer_count = 0
        for bs in block_structs:
            name = getattr(bs, "name", str(bs))
            if "refiner" in name.lower():
                refiner_count += 1
        
        print(f"  通常層数: {len(block_structs) - refiner_count}")
        print(f"  Refiner層数: {refiner_count}")
        
        if refiner_count >= 4:
            print("  SUCCESS: Refiner層がblock_structsに含まれています")
        else:
            print("  FAILURE: Refiner層がblock_structsに含まれていません")
            return False
        
        # 4. named_key_modules確認
        print("\n=== 4. Refiner関連モジュール確認 ===")
        refiner_modules = []
        for module_key, module_name, module, parent, field_name in model_struct.named_key_modules():
            if "refiner" in module_name.lower():
                refiner_modules.append(module_name)
        
        print(f"  Refiner関連モジュール数: {len(refiner_modules)}")
        if len(refiner_modules) > 0:
            print("  最初の5件:")
            for m in refiner_modules[:5]:
                print(f"    {m}")
            print("  SUCCESS: Refinerモジュールがnamed_key_modulesに含まれています")
        else:
            print("  FAILURE: Refinerモジュールがnamed_key_modulesに含まれていません")
            return False
        
        # クリーンアップ
        del pipeline, transformer, model_struct
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\n" + "=" * 80)
        print("事前検証完了: 全てのチェックに合格")
        print("量子化を実行できます")
        print("=" * 80)
        return True
        
    except Exception as e:
        import traceback
        print(f"  エラー: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
