import json
from evidence_cohort.contract import freeze

if __name__=='__main__':
    value=freeze();print(json.dumps({'status':value['status'],'execution':value['execution'],
                                  'clean_reconstruction':value['clean_reconstruction']},indent=2))
