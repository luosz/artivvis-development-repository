#include "JoesOGLRenderer.h"
#include "../TransferFunctionEditor/import_transfer_function_editor.h"

JoesOGLRenderer::JoesOGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera) : OpenGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera)
{
	visibilityHistogram.Init(screenWidth, screenHeight);

	_optimizer = new VisibilityTFOptimizer(&volume, &visibilityHistogram, &transferFunction);
	_intensityTFOptimizerV2 = new IntensityTFOptimizerV2(&volume, &transferFunction, &visibilityHistogram);
//	_optimizer = new RegionVisibilityOptimizer(&volume, &transferFunction, raycaster, &shaderManager, &camera);
//	_optimizer = new IntensityTFOptimizer(&volume, &transferFunction);

	//intensityOptimizerV2 = std::make_shared<IntensityTFOptimizerV2>(&volume, &transferFunction, &visibilityHistogram);
}


void JoesOGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	visibilityHistogram.CalculateHistogram(volume, transferFunction.tfTexture, shaderManager, camera);

	// if transfer function view is not NULL
	if (transferFunction.tfView)
	{
		// Ma's optimization
		if (transferFunction.tfView->isMaOptimizerEnable())
		{
			optimizer()->Optimize();
		}

		// Joe's intensity and visibility optimization
		if (transferFunction.tfView->isLuoOptimizerEnable())
		{
			transferFunction.tfView->updateTransferFunctionFromView();
			//auto p = dynamic_cast<TransferFunctionView *>(transferFunction.tfView);
			//if (p)
			//{
			//	auto optimizer2 = p->optimizer();
			//	optimizer2->BalanceVisibilityOnce();
			//}
			intensityTFOptimizerV2()->BalanceVisibilityOnce();
			transferFunction.tfView->updateViewFromTransferFunction();
		}
	}
	else
	{
		std::cout << "Warning: transferFunction.tfView is empty" << std::endl;
	}

	//if (visibilityHistogram.visibilityView)
	//{
	//	std::cout << "visibilityHistogram.visibilityView is not empty" << std::endl;
	//	visibilityHistogram.visibilityView->setVisibilityHistogram(visibilityHistogram.visibilities, visibilityHistogram.numVis);
	//	visibilityHistogram.visibilityView->draw();
	//}
	//else
	//{
	//	std::cout << "Warning: visibilityHistogram.visibilityView is empty" << std::endl;
	//}

	OpenGLRenderer::Draw(volume, shaderManager, camera);

//	if (optimizer())
//		optimizer()->Draw(shaderManager, camera);
//
//	visibilityHistogram.DrawHistogram(shaderManager, camera);
}
