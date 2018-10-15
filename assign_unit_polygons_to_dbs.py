# This is a one-off script for reading data from an Esri geodatabase, finding spatial polygons 
# (representing a drainage basin) with a certain percent overlap with another polygon layer and output the results to a simple csv file
#  Dorran Howell (2018)
#
# WORKFLOW:
#    Run Intersect between dbs and geo map
#	 Make sure Area field for intersect polygons is created
#	 Input relevant file/field names into this script
#	 Outputs a table

if __name__ == "__main__":
    import arcpy
    import os
 
    unit_polygons = "C:\\Users\\dorra\\Documents\\ETH\\Thesis\\Data\\Himalayas Main Dataset\\Himalayas_Master.gdb\\GeologicMapFeatures"
    tempgdb = "scratchpad.gdb"
    crs = ".\\HimalayaLCC.prj"
    unit_field = "unit_alias"
    
    threshold = 0.5
    
    intersect_file = "C:\\Users\\dorra\\Documents\\ETH\\Thesis\\ArcPro Projects\\scratchpad.gdb\\RawDBSearchResults_Intersect"
    
    if not os.path.exists(tempgdb):
        arcpy.CreateFileGDB_management('.\\',tempgdb)
        
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = crs
    
    
    
    
    ids = []
    aliases = []
    wholeareas = []
    intersectareas = []
    with arcpy.da.SearchCursor(intersect_file, ["DBID","unit_alias","Area","Shape_Area"]) as cursor:
        for row in cursor:
            ids.append(row[0])
            aliases.append(row[1])
            wholeareas.append(row[2])
            intersectareas.append(row[3])
            
            
    import pandas
    
    df = pandas.DataFrame({
            'id':ids,
            'alias':aliases,
            'whole':wholeareas,
            'intersect':intersectareas
            })
        
        
    
    grouped = df.groupby(['id','alias','whole']).sum()
    grouped.reset_index(inplace=True)
    grouped['frac'] = grouped.intersect/grouped.whole
    sort = grouped.sort_values(['id','frac'],ascending=False)
    sort = sort.reset_index(drop=True)
    out_ids = []
    out_alias = []
    out_frac = []
    for x in pandas.unique(sort['id']):
        rows = sort[sort['id']==x]
        rows = rows.reset_index(drop=True)
        check_perc = rows.frac > threshold
        if not any(check_perc):
            this_alias = 'MIX'
        else:
            this_alias = rows['alias'][0]
        out_ids.append(x)
        out_alias.append(this_alias)
        out_frac.append(rows['frac'][0])
        
    output = pandas.DataFrame({
            'DBID':out_ids,
            'unit_alias':out_alias,
            'unit_fraction':out_frac})
        
    output.sort_values('DBID',inplace=True)
    output.to_csv('.\\ZoneAssignments_07102018.csv')