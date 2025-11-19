from app.utils.state_utils import set_running
from app.services.supabase_service import fetch_pending_classification, update_classification
from app.utils.image_utils import analysis_image


def analysis():
    while True:
        classification = fetch_pending_classification()
        if classification == None:
            break

        result_image_path, analysis = analysis_image(classification["image"])
        update_classification(classification['id'], analysis, result_image_path)
    
    set_running(False)