CREATE TABLE TEMP as SELECT A.ChooseModel, A.Camera, B. Weights, A.Confidence, A.OK_Cam1, A.OK_Cam2, A.OK_Cam3, A.NG_Cam1, A.NG_Cam2, A.NG_Cam3, A.Folder_OK_Cam1, A.Folder_OK_Cam2, A.Folder_OK_Cam3, A.Folder_NG_Cam1, A.Folder_NG_Cam2, A.Folder_NG_Cam3, A.Joined, A.Ok, A.Num, A.NG, A.WidthMin, A. WidthMax, A.HeightMin, A.HeightMax, A.PLC_NG, A.PLC_OK FROM BKMODEL as A INNER JOIN MYMODEL as B ON A.ChooseModel = B.ChooseModel And A.Camera = B.Camera