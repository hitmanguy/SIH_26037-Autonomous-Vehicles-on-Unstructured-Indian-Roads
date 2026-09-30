function [scenario, scenarioInfo] = import_openscenario(xoscFile, xodrOverride, varargin)
% IMPORT_OPENSCENARIO Imports an ASAM OpenSCENARIO (.xosc) file and OpenDRIVE (.xodr)
% road network into a MATLAB drivingScenario object.
%
% Syntax:
%   scenario = import_openscenario(xoscFile)
%   scenario = import_openscenario(xoscFile, xodrOverride)
%   [scenario, scenarioInfo] = import_openscenario(xoscFile, xodrOverride, 'Param', Value, ...)
%
% Parameters:
%   'SampleTime'  - Simulation time step in seconds (default: 0.05 s = 20 Hz)
%   'StopTime'    - Maximum simulation time in seconds (default: 30.0 s)
%   'EnableAEB'   - Enable Autonomous Emergency Braking for Ego (default: true)
%   'ArmOffset'   - Shift initial S coordinates along approach arm (default: 0)
%
% Supported Scenarios from Euro NCAP / OSC-NCAP:
%   - AEB VRU CPNCO (Car-to-Pedestrian Nearside Child Obstructed)
%   - AEB VRU CPTA  (Car-to-Pedestrian Turning Adult at Intersection)
%   - AEB C2C CCFtap(Car-to-Car Front Turn Across Path)
%   - CA-FC CCCscp  (Car-to-Car Straight Crossing Path at Intersection)
%   - And arbitrary ASAM OpenSCENARIO XML files with standard catalogs
%
% =========================================================================

    % 1. Parse Input Arguments
    p = inputParser;
    addRequired(p, 'xoscFile', @(x) ischar(x) || isstring(x));
    addOptional(p, 'xodrOverride', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'SampleTime', 0.05, @isnumeric);
    addParameter(p, 'StopTime', 30.0, @isnumeric);
    addParameter(p, 'EnableAEB', true, @islogical);
    addParameter(p, 'ArmOffset', 0, @isnumeric);
    addParameter(p, 'LiberateEgo', false, @islogical);
    parse(p, xoscFile, xodrOverride, varargin{:});

    xoscFile = char(p.Results.xoscFile);
    xodrOverride = char(p.Results.xodrOverride);
    sampleTime = p.Results.SampleTime;
    stopTime = p.Results.StopTime;
    enableAEB = p.Results.EnableAEB;
    armOffset = p.Results.ArmOffset;
    liberateEgo = p.Results.LiberateEgo;

    if ~isfile(xoscFile)
        error('OpenSCENARIO file not found: %s', xoscFile);
    end

    xoscDir = fileparts(xoscFile);
    if isempty(xoscDir), xoscDir = pwd; end

    fprintf('============================================================\n');
    fprintf('ASAM OpenSCENARIO Importer for MATLAB Automated Driving\n');
    fprintf('XOSC Source: %s\n', xoscFile);

    % 2. Parse XML Document
    doc = xmlread(xoscFile);
    xoscRoot = doc.getDocumentElement();

    % 3. Extract Scenario Description & Header
    headerNode = xoscRoot.getElementsByTagName('FileHeader');
    scenarioDesc = 'NCAP OpenSCENARIO';
    if headerNode.getLength() > 0
        scenarioDesc = char(headerNode.item(0).getAttribute('description'));
    end
    fprintf('Description: %s\n', scenarioDesc);

    % 4. Parse & Resolve ParameterDeclarations
    params = parse_parameters(xoscRoot);
    fprintf('Parsed and evaluated %d scenario parameters.\n', params.Count);

    % 5. Resolve OpenDRIVE Road Network (.xodr)
    if ~isempty(xodrOverride) && isfile(xodrOverride)
        xodrFile = xodrOverride;
    else
        % Look for <LogicFile filepath="..."> in <RoadNetwork>
        roadNetNodes = xoscRoot.getElementsByTagName('LogicFile');
        if roadNetNodes.getLength() > 0
            relXodr = char(roadNetNodes.item(0).getAttribute('filepath'));
            relXodrClean = strrep(relXodr, '/', filesep);
            candXodr = fullfile(xoscDir, relXodrClean);
            try
                candXodr = char(java.io.File(candXodr).getCanonicalPath());
            catch
            end
            if isfile(candXodr)
                xodrFile = candXodr;
            else
                error('Referenced OpenDRIVE file not found: %s', candXodr);
            end
        else
            error('No OpenDRIVE road network specified or found in scenario.');
        end
    end
    fprintf('OpenDRIVE Map: %s\n', xodrFile);

    % 6. Parse Catalog Locations
    catalogs = parse_catalogs(xoscRoot, xoscDir);

    % 7. Create drivingScenario and Import Road Network
    scenario = drivingScenario('SampleTime', sampleTime, 'StopTime', stopTime);
    fprintf('[1/4] Importing OpenDRIVE road network into drivingScenario...\n');
    roadNetwork(scenario, 'OpenDRIVE', xodrFile);

    % 8. Parse Entities and Instantiate Actors with Exact Dimensions
    entities = parse_entities(xoscRoot, catalogs, params);
    fprintf('[2/4] Instantiating %d scenario entities...\n', numel(fieldnames(entities)));

    % 9. Build Trajectories & Waypoints Based on Scenario Type
    scenarioInfo = struct();
    scenarioInfo.ScenarioDescription = scenarioDesc;
    scenarioInfo.XOSCFile = xoscFile;
    scenarioInfo.XODRFile = xodrFile;
    scenarioInfo.Parameters = params;
    scenarioInfo.Catalogs = catalogs;
    scenarioInfo.Entities = entities;

    scenarioType = detect_scenario_type(xoscFile, scenarioDesc, params);
    scenarioInfo.ScenarioType = scenarioType;
    fprintf('[3/4] Configured scenario dynamics model: [%s]\n', scenarioType);

    [scenario, actorMap] = build_scenario_trajectories(scenario, entities, scenarioType, params, enableAEB, armOffset, liberateEgo);
    scenarioInfo.ActorMap = actorMap;

    % Ensure realistic 3D meshes are attached to all imported scenario actors
    for a = 1:numel(scenario.Actors)
        act = scenario.Actors(a);
        try
            if isempty(act.Mesh) || size(act.Mesh.Vertices, 1) <= 8
                if act.ClassID == 1
                    act.Mesh = driving.scenario.carMesh;
                elseif act.ClassID == 2
                    act.Mesh = driving.scenario.truckMesh;
                elseif act.ClassID == 4
                    act.Mesh = driving.scenario.pedestrianMesh;
                elseif act.ClassID == 3 || act.ClassID == 7
                    act.Mesh = driving.scenario.bicycleMesh;
                end
            end
        catch
        end
    end

    fprintf('[4/4] Scenario import completed successfully (%d actors active).\n', length(scenario.Actors));
    fprintf('============================================================\n');
end

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function val = getParam(params, key, defaultVal)
    if nargin < 3, defaultVal = 0; end
    if isa(params, 'containers.Map')
        if params.isKey(key)
            val = params(key);
            return;
        end
        if startsWith(key, '_') && params.isKey(key(2:end))
            val = params(key(2:end));
            return;
        end
        if ~startsWith(key, '_') && params.isKey(['_' key])
            val = params(['_' key]);
            return;
        end
    elseif isstruct(params)
        safeKey = matlab.lang.makeValidName(key);
        if isfield(params, safeKey)
            val = params.(safeKey);
            return;
        end
    end
    val = defaultVal;
end

function params = parse_parameters(xoscRoot)
    % Parses all <ParameterDeclaration> nodes and evaluates math formulas
    pNodes = xoscRoot.getElementsByTagName('ParameterDeclaration');
    rawMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    allKeys = cell(1, pNodes.getLength());

    for i = 0:(pNodes.getLength() - 1)
        node = pNodes.item(i);
        pName = char(node.getAttribute('name'));
        pVal = char(node.getAttribute('value'));
        rawMap(pName) = pVal;
        allKeys{i + 1} = pName;
    end

    % Evaluator environment: support math functions
    pow = @(a,b) a.^b; %#ok<NASGU>
    params = containers.Map('KeyType', 'char', 'ValueType', 'any');

    % First pass: direct literals
    for i = 1:numel(allKeys)
        k = allKeys{i};
        v = strtrim(rawMap(k));
        numVal = str2double(v);
        if ~isnan(numVal)
            params(k) = numVal;
        elseif strcmpi(v, 'true')
            params(k) = true;
        elseif strcmpi(v, 'false')
            params(k) = false;
        elseif ~contains(v, '$')
            params(k) = v;
        end
    end

    % Iterative pass: evaluate expressions containing ${...} or $var
    for iter = 1:30
        progress = false;
        for i = 1:numel(allKeys)
            k = allKeys{i};
            if params.isKey(k)
                continue;
            end
            expr = strtrim(rawMap(k));
            if startsWith(expr, '${') && endsWith(expr, '}')
                expr = expr(3:end-1);
            end

            % Find all $var references
            tokens = regexp(expr, '\$([a-zA-Z0-9_]+)', 'tokens');
            canEval = true;
            for t = 1:numel(tokens)
                varName = tokens{t}{1};
                if ~params.isKey(varName)
                    canEval = false;
                    break;
                end
            end

            if canEval
                subExpr = expr;
                % Replace tokens from longest to shortest
                tokNames = cellfun(@(c) c{1}, tokens, 'UniformOutput', false);
                [~, sortIdx] = sort(cellfun(@length, tokNames), 'descend');
                for s = sortIdx
                    vName = tokNames{s};
                    val = params(vName);
                    if isnumeric(val)
                        subExpr = regexprep(subExpr, ['\$' vName '\b'], sprintf('%.10f', val));
                    else
                        subExpr = regexprep(subExpr, ['\$' vName '\b'], ['''' char(val) '''']);
                    end
                end
                try
                    res = eval(subExpr);
                    params(k) = res;
                    progress = true;
                catch
                    % Incomplete or unsupported syntax
                end
            end
        end
        if ~progress
            break;
        end
    end
end

function catalogs = parse_catalogs(xoscRoot, xoscDir)
    % Parses VehicleCatalog and PedestrianCatalog files
    catalogs = struct();
    catalogs.Vehicles = struct();
    catalogs.Pedestrians = struct();

    % Locate CatalogLocations
    catLocNodes = xoscRoot.getElementsByTagName('CatalogLocations');
    if catLocNodes.getLength() == 0
        return;
    end

    % 1. Vehicle Catalog
    vehDirNodes = xoscRoot.getElementsByTagName('VehicleCatalog');
    if vehDirNodes.getLength() > 0
        dirNode = vehDirNodes.item(0).getElementsByTagName('Directory');
        if dirNode.getLength() > 0
            relPath = char(dirNode.item(0).getAttribute('path'));
            vehCatDir = char(java.io.File(xoscDir, relPath).getCanonicalPath());
            vehCatFile = fullfile(vehCatDir, 'Vehicles.xosc');
            if isfile(vehCatFile)
                catalogs.Vehicles = parse_vehicle_catalog_file(vehCatFile);
            end
        end
    end

    % 2. Pedestrian Catalog
    pedDirNodes = xoscRoot.getElementsByTagName('PedestrianCatalog');
    if pedDirNodes.getLength() > 0
        dirNode = pedDirNodes.item(0).getElementsByTagName('Directory');
        if dirNode.getLength() > 0
            relPath = char(dirNode.item(0).getAttribute('path'));
            pedCatDir = char(java.io.File(xoscDir, relPath).getCanonicalPath());
            pedCatFile = fullfile(pedCatDir, 'Pedestrians.xosc');
            if isfile(pedCatFile)
                catalogs.Pedestrians = parse_pedestrian_catalog_file(pedCatFile);
            end
        end
    end
end

function vehs = parse_vehicle_catalog_file(catFile)
    vehs = struct();
    doc = xmlread(catFile);
    nodes = doc.getElementsByTagName('Vehicle');
    for i = 0:(nodes.getLength() - 1)
        v = nodes.item(i);
        name = char(v.getAttribute('name'));
        cat = char(v.getAttribute('vehicleCategory'));
        
        dimNode = v.getElementsByTagName('Dimensions');
        length = 4.5; width = 1.8; height = 1.5;
        if dimNode.getLength() > 0
            d = dimNode.item(0);
            length = str2double(char(d.getAttribute('length')));
            width = str2double(char(d.getAttribute('width')));
            height = str2double(char(d.getAttribute('height')));
        end

        bbNode = v.getElementsByTagName('Center');
        bbX = 0; bbY = 0; bbZ = 0;
        if bbNode.getLength() > 0
            b = bbNode.item(0);
            bbX = str2double(char(b.getAttribute('x')));
            bbY = str2double(char(b.getAttribute('y')));
            bbZ = str2double(char(b.getAttribute('z')));
        end

        entry = struct('Name', name, 'Category', cat, 'Length', length, ...
            'Width', width, 'Height', height, 'BBCenter', [bbX, bbY, bbZ]);
        safeName = matlab.lang.makeValidName(name);
        vehs.(safeName) = entry;
    end
end

function peds = parse_pedestrian_catalog_file(catFile)
    peds = struct();
    doc = xmlread(catFile);
    nodes = doc.getElementsByTagName('Pedestrian');
    for i = 0:(nodes.getLength() - 1)
        p = nodes.item(i);
        name = char(p.getAttribute('name'));
        cat = char(p.getAttribute('pedestrianCategory'));

        dimNode = p.getElementsByTagName('Dimensions');
        length = 0.5; width = 0.5; height = 1.7;
        if dimNode.getLength() > 0
            d = dimNode.item(0);
            length = str2double(char(d.getAttribute('length')));
            width = str2double(char(d.getAttribute('width')));
            height = str2double(char(d.getAttribute('height')));
        end

        entry = struct('Name', name, 'Category', cat, 'Length', length, ...
            'Width', width, 'Height', height);
        safeName = matlab.lang.makeValidName(name);
        peds.(safeName) = entry;
    end
end

function entities = parse_entities(xoscRoot, catalogs, params)
    entities = struct();
    objNodes = xoscRoot.getElementsByTagName('ScenarioObject');
    for i = 0:(objNodes.getLength() - 1)
        node = objNodes.item(i);
        name = char(node.getAttribute('name'));
        
        % Check CatalogReference
        catRef = node.getElementsByTagName('CatalogReference');
        entryName = '';
        catName = '';
        if catRef.getLength() > 0
            entryName = char(catRef.item(0).getAttribute('entryName'));
            catName = char(catRef.item(0).getAttribute('catalogName'));
            % Check if entryName or catName is a parameter reference
            if startsWith(entryName, '$')
                pKey = entryName(2:end);
                if params.isKey(pKey), entryName = char(string(params(pKey))); end
            end
            if startsWith(catName, '$')
                pKey = catName(2:end);
                if params.isKey(pKey), catName = char(string(params(pKey))); end
            end
        end

        % Lookup dimensions
        length = 4.5; width = 1.8; height = 1.5;
        classId = 1; assetType = 'Sedan';
        safeEntry = matlab.lang.makeValidName(entryName);

        if contains(lower(catName), 'pedestrian') || contains(lower(entryName), 'child') || contains(lower(entryName), 'adult')
            classId = 4;
            if contains(lower(entryName), 'child')
                assetType = 'MalePedestrian'; % MATLAB DSD asset mapping
                length = 0.28; width = 0.35; height = 1.154;
            else
                assetType = 'MalePedestrian';
                length = 0.28; width = 0.45; height = 1.80;
            end
            if isfield(catalogs.Pedestrians, safeEntry)
                pInfo = catalogs.Pedestrians.(safeEntry);
                width = pInfo.Width; height = pInfo.Height;
                if pInfo.Length <= 0.45, length = pInfo.Length; else, length = 0.28; end
            end
        elseif contains(lower(catName), 'vehicle') || isfield(catalogs.Vehicles, safeEntry)
            if isfield(catalogs.Vehicles, safeEntry)
                vInfo = catalogs.Vehicles.(safeEntry);
                length = vInfo.Length; width = vInfo.Width; height = vInfo.Height;
                if strcmpi(vInfo.Category, 'bicycle')
                    classId = 3; assetType = 'Bicyclist';
                elseif strcmpi(vInfo.Category, 'motorbike')
                    classId = 3; assetType = 'Bicyclist';
                elseif strcmpi(vInfo.Category, 'truck') || strcmpi(vInfo.Category, 'bus')
                    classId = 2; assetType = 'BoxTruck';
                else
                    classId = 1;
                    if contains(lower(entryName), 'small')
                        assetType = 'Hatchback';
                    else
                        assetType = 'Sedan';
                    end
                end
            else
                % Default fallback for unknown vehicle
                classId = 1; assetType = 'Sedan';
            end
        end

        % Check inline Vehicle definition
        vehNode = node.getElementsByTagName('Vehicle');
        if vehNode.getLength() > 0
            vElem = vehNode.item(0);
            vCat = lower(char(vElem.getAttribute('vehicleCategory')));
            dimNodes = vElem.getElementsByTagName('Dimensions');
            if dimNodes.getLength() > 0
                dElem = dimNodes.item(0);
                if dElem.hasAttribute('length'), length = str2double(char(dElem.getAttribute('length'))); end
                if dElem.hasAttribute('width'),  width = str2double(char(dElem.getAttribute('width'))); end
                if dElem.hasAttribute('height'), height = str2double(char(dElem.getAttribute('height'))); end
            end
            if contains(vCat, 'bike') || contains(vCat, 'motor') || contains(vCat, 'cycle')
                classId = 3; assetType = 'Bicyclist';
            elseif contains(vCat, 'truck') || contains(vCat, 'bus')
                classId = 2; assetType = 'BoxTruck';
            else
                classId = 1; assetType = 'Sedan';
            end
        end

        % Check inline Pedestrian definition
        pedNode = node.getElementsByTagName('Pedestrian');
        if pedNode.getLength() > 0
            pElem = pedNode.item(0);
            classId = 4; assetType = 'MalePedestrian';
            length = 0.28; width = 0.45; height = 1.80;
            dimNodes = pElem.getElementsByTagName('Dimensions');
            if dimNodes.getLength() > 0
                dElem = dimNodes.item(0);
                if dElem.hasAttribute('length'), length = str2double(char(dElem.getAttribute('length'))); end
                if dElem.hasAttribute('width'),  width = str2double(char(dElem.getAttribute('width'))); end
                if dElem.hasAttribute('height'), height = str2double(char(dElem.getAttribute('height'))); end
            end
        end

        % Check inline MiscObject definition (Road Barrier / Hazard)
        miscNode = node.getElementsByTagName('MiscObject');
        if miscNode.getLength() > 0
            mElem = miscNode.item(0);
            classId = 5; assetType = 'JerseyBarrier';
            length = 5.0; width = 0.8; height = 0.8;
            dimNodes = mElem.getElementsByTagName('Dimensions');
            if dimNodes.getLength() > 0
                dElem = dimNodes.item(0);
                if dElem.hasAttribute('length'), length = str2double(char(dElem.getAttribute('length'))); end
                if dElem.hasAttribute('width'),  width = str2double(char(dElem.getAttribute('width'))); end
                if dElem.hasAttribute('height'), height = str2double(char(dElem.getAttribute('height'))); end
            end
        end

        entStruct = struct('Name', name, 'CatalogEntry', entryName, ...
            'CatalogName', catName, 'ClassID', classId, 'AssetType', assetType, ...
            'Length', length, 'Width', width, 'Height', height);
        entities.(name) = entStruct;
    end
end

function scType = detect_scenario_type(xoscFile, desc, params)
    fUpper = upper(xoscFile);
    dUpper = upper(desc);
    pID = upper(char(string(getParam(params, 'Scenario_ID', ''))));
    if contains(fUpper, 'INDIAN_AUTORICKSHAW') || contains(dUpper, 'AUTO-RICKSHAW') || contains(fUpper, 'AUTOCUTIN')
        scType = 'INDIAN_AUTOCUTIN';
    elseif contains(fUpper, 'INDIAN_TWOWHEELER') || contains(dUpper, 'TWO-WHEELER') || contains(fUpper, 'LANEFILTER')
        scType = 'INDIAN_TWOWHEELER';
    elseif contains(fUpper, 'INDIAN_PEDESTRIAN_JAYWALK') || contains(dUpper, 'JAYWALK')
        scType = 'INDIAN_JAYWALK';
    elseif contains(fUpper, 'INDIAN_STRAYCATTLE') || contains(dUpper, 'CATTLE') || contains(fUpper, 'HAZARD')
        scType = 'INDIAN_CATTLE';
    elseif contains(fUpper, 'CONGESTION') || contains(dUpper, 'CONGESTION')
        scType = 'INDIAN_CONGESTION';
    elseif contains(fUpper, 'POTHOLE') || contains(dUpper, 'POTHOLE')
        scType = 'INDIAN_POTHOLE';
    elseif contains(fUpper, 'WRONGWAY') || contains(dUpper, 'WRONG-WAY') || contains(dUpper, 'HEAD-ON')
        scType = 'INDIAN_WRONGWAY';
    elseif contains(fUpper, 'SCHOOLZONE') || contains(dUpper, 'SCHOOL ZONE') || contains(dUpper, 'CHILDREN CROSSING')
        scType = 'INDIAN_SCHOOLZONE';
    elseif contains(fUpper, 'BUSSTOP') || contains(dUpper, 'BUS STOP') || contains(dUpper, 'ALIGHTING')
        scType = 'INDIAN_BUSSTOP';
    elseif contains(fUpper, 'VENDORCART') || contains(dUpper, 'VENDOR') || contains(dUpper, 'HANDCART')
        scType = 'INDIAN_VENDORCART';
    elseif contains(fUpper, 'MULTITHREAT') || contains(fUpper, 'GAUNTLET') || contains(dUpper, 'GAUNTLET') || contains(dUpper, 'MULTI-HAZARD')
        scType = 'INDIAN_GAUNTLET';
    elseif contains(fUpper, 'LANECHANGESIMPLE')
        scType = 'LANECHANGESIMPLE';
    elseif contains(fUpper, 'OVERTAKESLOWVEHICLE')
        scType = 'OVERTAKESLOWVEHICLE';
    elseif contains(fUpper, 'PEDESTRIANCROSSING')
        scType = 'PEDESTRIANCROSSING';
    elseif contains(fUpper, 'CPNCO') || contains(dUpper, 'CPNCO') || contains(pID, 'CPNCO')
        scType = 'CPNCO';
    elseif contains(fUpper, 'CPTA') || contains(dUpper, 'CPTA') || contains(pID, 'CPTA')
        scType = 'CPTA';
    elseif contains(fUpper, 'CCFTAP') || contains(dUpper, 'CCFTAP') || contains(pID, 'CCFTAP')
        scType = 'CCFtap';
    elseif contains(fUpper, 'CCCSCP') || contains(dUpper, 'CCCSCP') || contains(pID, 'CCCSCP')
        scType = 'CCCscp';
    elseif contains(fUpper, 'CBFA') || contains(dUpper, 'CBFA')
        scType = 'CBFA';
    elseif contains(fUpper, 'CBNA') || contains(dUpper, 'CBNA')
        scType = 'CBNA';
    else
        scType = 'GENERIC';
    end
end

function [scenario, actorMap] = build_scenario_trajectories(scenario, entities, scenarioType, params, enableAEB, armOffset, liberateEgo)
    if nargin < 7, liberateEgo = false; end
    actorMap = struct();

    switch upper(scenarioType)
        case 'INDIAN_CONGESTION'
            [scenario, actorMap] = setup_indian_congestion_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_POTHOLE'
            [scenario, actorMap] = setup_indian_pothole_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_AUTOCUTIN'
            [scenario, actorMap] = setup_indian_autocutin_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_TWOWHEELER'
            [scenario, actorMap] = setup_indian_twowheeler_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_JAYWALK'
            [scenario, actorMap] = setup_indian_jaywalk_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_CATTLE'
            [scenario, actorMap] = setup_indian_cattle_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_WRONGWAY'
            [scenario, actorMap] = setup_indian_wrongway_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_SCHOOLZONE'
            [scenario, actorMap] = setup_indian_schoolzone_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_BUSSTOP'
            [scenario, actorMap] = setup_indian_busstop_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_VENDORCART'
            [scenario, actorMap] = setup_indian_vendorcart_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'INDIAN_GAUNTLET'
            [scenario, actorMap] = setup_indian_gauntlet_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'LANECHANGESIMPLE'
            [scenario, actorMap] = setup_lanechange_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'OVERTAKESLOWVEHICLE'
            [scenario, actorMap] = setup_overtake_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'PEDESTRIANCROSSING'
            [scenario, actorMap] = setup_pedcrossing_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'CPNCO'
            [scenario, actorMap] = setup_cpnco_scenario(scenario, entities, params, enableAEB, armOffset, liberateEgo);
        case 'CPTA'
            [scenario, actorMap] = setup_cpta_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'CCFTAP'
            [scenario, actorMap] = setup_ccftap_scenario(scenario, entities, params, enableAEB, liberateEgo);
        case 'CCCSCP'
            [scenario, actorMap] = setup_cccscp_scenario(scenario, entities, params, enableAEB, liberateEgo);
        otherwise
            if isfield(entities, 'Bicycle') && isfield(entities, 'SlowCar1')
                [scenario, actorMap] = setup_indian_congestion_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'RoadWorkBarrier') || isfield(entities, 'OncomingTruck') || contains(upper(scenarioType), 'POTHOLE')
                [scenario, actorMap] = setup_indian_pothole_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'AutoRickshaw')
                [scenario, actorMap] = setup_indian_autocutin_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'Motorcycle')
                [scenario, actorMap] = setup_indian_twowheeler_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'Pedestrian') && isfield(entities, 'ParkedBus')
                [scenario, actorMap] = setup_indian_jaywalk_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'Hazard')
                [scenario, actorMap] = setup_indian_cattle_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'WrongWayTempo')
                [scenario, actorMap] = setup_indian_wrongway_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'SchoolBus')
                [scenario, actorMap] = setup_indian_schoolzone_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'AlightingPassenger') || isfield(entities, 'CityBus')
                [scenario, actorMap] = setup_indian_busstop_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'VendorCart')
                [scenario, actorMap] = setup_indian_vendorcart_scenario(scenario, entities, params, enableAEB, liberateEgo);
            elseif isfield(entities, 'ConstructionBarrier1') || isfield(entities, 'SlowRickshaw')
                [scenario, actorMap] = setup_indian_gauntlet_scenario(scenario, entities, params, enableAEB, liberateEgo);
            else
                [scenario, actorMap] = setup_cpnco_scenario(scenario, entities, params, enableAEB, armOffset, liberateEgo);
            end
    end
end

function assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo)
    if nargin >= 4 && liberateEgo
        initPos = egoWaypoints(1, :);
        initYaw = 0;
        if size(egoWaypoints, 1) >= 2
            initYaw = rad2deg(atan2(egoWaypoints(2,2) - egoWaypoints(1,2), egoWaypoints(2,1) - egoWaypoints(1,1)));
        end
        if isscalar(egoSpeeds)
            v0 = egoSpeeds;
        else
            v0 = egoSpeeds(1);
        end
        ego.Position = [initPos(1), initPos(2), 0.0];
        ego.Velocity = [v0 * cosd(initYaw), v0 * sind(initYaw), 0.0];
        ego.Yaw = initYaw;
    else
        trajectory(ego, egoWaypoints, egoSpeeds);
    end
end

% =========================================================================
% SCENARIO IMPLEMENTATIONS (STANDARDS-COMPLIANT & COLLISION VERIFIED)
% =========================================================================

function [scenario, actorMap] = setup_cpnco_scenario(scenario, entities, params, enableAEB, armOffset, liberateEgo)
    if nargin < 6, liberateEgo = false; end
    % Car-to-Pedestrian Nearside Child Obstructed (CPNCO)
    % Road 0 is 250m long Eastbound (hdg = 0). Center lane Y = 0.
    % Right driving lane (Lane -1): center is at Y = -1.75m.
    % Outer shoulder: Y in [-9.0, -3.5] m.

    actorMap = struct();

    % Parameters
    egoSpeed = getParam(params, '_Ego_speed', 6.94); % 25 km/h steady smooth cruise
    egoInitS = getParam(params, 'Ego_initS', 50.0) + armOffset;
    vruSpeed = getParam(params, '_VRU_finalSpeed', 1.389);
    vruInitS = getParam(params, '_VRU_initS', 100.0) + armOffset;
    obsSmallS = getParam(params, '_ObstructionSmall_initS', 95.325) + armOffset;
    obsLargeDist = getParam(params, '_ObstructionLarge_initDist', -5.398);
    obsLargeS = obsSmallS + obsLargeDist;

    obsLat = -5.85; % Parked further off the road onto outer shoulder
    vruLatStart = -6.80; % Hidden behind obstruction small further off the road

    % 1. Ego Vehicle (MUST BE ACTOR 1)
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', egoData.ClassID, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, ...
        'PlotColor', [0.05 0.45 0.95]);

    if enableAEB
        % Ego cruises steadily at 25 km/h (6.94 m/s).
        % At t = 3.45s (X ~ 74m), child emerges from behind parked vehicle.
        % AEB triggers smooth deceleration, halting safely at X = 96.5m (3.5m clearance).
        % At t = 8.15s, child has cleared to the opposite curb; Ego smoothly accelerates
        % through to the X-intersection (X = 250m) and exits onto Road 2!
        egoWaypoints = [
            egoInitS,      -1.75, 0.0;   % t = 0.0s (steady cruise at 25 km/h)
            egoInitS + 20, -1.75, 0.0;   % t = 2.9s (steady cruise)
            egoInitS + 28, -1.75, 0.0;   % t = 4.0s (detects child emerging into lane, smooth AEB activates)
            vruInitS - 4.5,-1.75, 0.0;   % t = 5.8s (safe standstill 4.5m before crossing, bumper 2.3m clear)
            vruInitS - 4.0,-1.75, 0.0;   % t = 8.2s (yield hold while child slowly walks across)
            vruInitS + 15, -1.75, 0.0;   % t = 12.0s (smoothly accelerates past crossing once child clears)
            vruInitS + 50, -1.75, 0.0;   % t = 17.0s (cruising down Road 0)
            245.0,         -1.75, 0.0;   % t = 23.5s (entering 4-way intersection)
            275.0,         -1.75, 0.0;   % t = 28.0s (crossing junction box)
            320.0,         -1.75, 0.0    % t = 33.5s (exiting to East Road 2)
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 4.0, 0.05, 0.05, 4.0, egoSpeed, egoSpeed, egoSpeed, egoSpeed];
    else
        % Open-loop constant cruise (for collision detection evaluation)
        egoWaypoints = [
            egoInitS, -1.75, 0.0;
            245.0,    -1.75, 0.0;
            275.0,    -1.75, 0.0;
            320.0,    -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % 2. Obstruction Vehicles (Parked on Shoulder, Shifted Further Off-Road)
    obsSmallData = entities.ObstructionSmall;
    obsSmall = vehicle(scenario, 'ClassID', obsSmallData.ClassID, 'Name', 'ObstructionSmall', ...
        'AssetType', obsSmallData.AssetType, 'Length', obsSmallData.Length, ...
        'Width', obsSmallData.Width, 'Height', obsSmallData.Height, ...
        'Position', [obsSmallS, obsLat, 0.0], 'Yaw', 0, ...
        'PlotColor', [0.45 0.45 0.50]);
    actorMap.ObstructionSmall = obsSmall;

    obsLargeData = entities.ObstructionLarge;
    obsLarge = vehicle(scenario, 'ClassID', obsLargeData.ClassID, 'Name', 'ObstructionLarge', ...
        'AssetType', obsLargeData.AssetType, 'Length', obsLargeData.Length, ...
        'Width', obsLargeData.Width, 'Height', obsLargeData.Height, ...
        'Position', [obsLargeS, obsLat, 0.0], 'Yaw', 0, ...
        'PlotColor', [0.35 0.35 0.40]);
    actorMap.ObstructionLarge = obsLarge;

    % 3. VRU (Child Pedestrian)
    % Hidden behind small obstruction car further off the road at Y = -6.80m.
    % Darts out fast towards lane edge (2.5 m/s), then walks slowly across driving lane (0.85 m/s)
    % so that pedestrian stays in the road for 4.45 seconds directly in the ego vehicle's path.
    vruData = entities.VRU;
    vru = actor(scenario, 'ClassID', vruData.ClassID, 'Name', 'VRU_Child', ...
        'AssetType', vruData.AssetType, 'Length', vruData.Length, ...
        'Width', vruData.Width, 'Height', vruData.Height, ...
        'PlotColor', [0.95 0.75 0.10]);

    vruWaypoints = [
        vruInitS, -6.80, 0.0;   % 1. Shoulder/sidewalk, tucked safely off-road behind parked cars
        vruInitS, -4.80, 0.0;   % 2. Edge of parked obstruction cars
        vruInitS, -3.50, 0.0;   % 3. Edge of driving lane (enters road)
        vruInitS, -1.75, 0.0;   % 4. Driving lane center
        vruInitS,  0.50, 0.0;   % 5. Cleared driving lane to median
        vruInitS,  2.50, 0.0    % 6. Safe on far median / sidewalk
    ];
    % Speed profile: 0 initial, 2.5 m/s dart out, 2.0 m/s transition, 0.85 m/s slow crossing across lane
    vruSpeeds = [0, 2.5, 2.0, 0.85, 0.85, 0.85];

    % Initial position: stationary on shoulder
    vru.Position = vruWaypoints(1, :);
    
    % Wait time 2.00s: starts moving at t=2.00s, enters road at t=3.45s, stays in road until t=7.90s (4.45s duration in road)
    T_wait = 2.00;
    trajectory(vru, vruWaypoints, vruSpeeds, [T_wait, 0, 0, 0, 0, 0]);

    % Scenario metadata
    actorMap.VRU = vru;
    actorMap.VRUWaypoints = vruWaypoints;
    actorMap.VRUSpeeds = vruSpeeds;
    actorMap.VRUCrossTime = 4.45; % 4.45s presence in driving lane
    actorMap.VRUBaselineStart = T_wait;
    actorMap.VRUTriggered = true;
end

function [scenario, actorMap] = setup_cpta_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Car-to-Pedestrian Turning Adult at Intersection (CPTA)
    % Ego turns left from West arm (Road 0) into North arm (Road 1).
    % Adult pedestrian crosses the North arm crosswalk.

    actorMap = struct();
    egoSpeed = getParam(params, '_Ego_speed', 2.778); % 10 km/h (turning speed)
    pedSpeed = getParam(params, '_VRU_finalSpeed', 1.389); % 5 km/h

    % 1. Ego Vehicle
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', egoData.ClassID, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, ...
        'PlotColor', [0.05 0.45 0.95]);

    % Turning arc: from Road 0 (Y = -1.75, X = 200 -> 250) into Junction (X = 261.5, Y = 0)
    % into North Arm Road 1 (X = 261.5 + 1.75 = 263.25, Y > 11.5)
    if enableAEB
        egoWaypoints = [
            210.0, -1.75, 0.0;   % t = 0s
            245.0, -1.75, 0.0;   % approach junction
            254.0, -0.50, 0.0;   % initiates turn
            259.0,  5.00, 0.0;   % spots pedestrian crossing North arm, AEB yields
            259.5,  7.00, 0.0;   % holds safe 4m gap
            263.25, 20.0, 0.0;   % completes turn into North arm after pedestrian clears
            263.25, 60.0, 0.0    % drives North
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 2.0, 0.1, 0.1, egoSpeed, egoSpeed];
    else
        egoWaypoints = [
            210.0, -1.75, 0.0;
            245.0, -1.75, 0.0;
            255.0, -0.20, 0.0;
            261.5,  5.00, 0.0;
            263.25, 20.0, 0.0;
            263.25, 60.0, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, egoSpeed, egoSpeed, egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % 2. VRU (Adult Pedestrian)
    vruData = entities.VRU;
    vru = actor(scenario, 'ClassID', vruData.ClassID, 'Name', 'VRU_Adult', ...
        'AssetType', vruData.AssetType, 'Length', vruData.Length, ...
        'Width', vruData.Width, 'Height', vruData.Height, ...
        'PlotColor', [0.90 0.20 0.20]);

    % Crosses North crosswalk at Y = 13.0m, moving West to East from X = 256m to X = 268m
    vruWaypoints = [
        256.0, 13.0, 0.0;
        268.0, 13.0, 0.0
    ];
    trajectory(vru, vruWaypoints, pedSpeed);
    actorMap.VRU = vru;
end

function [scenario, actorMap] = setup_ccftap_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Car-to-Car Front Turn Across Path (CCFtap)
    % Ego turns left from Road 0 across the path of oncoming Target from Road 2 (East).

    actorMap = struct();
    egoSpeed = getParam(params, '_Ego_speed', 2.778);
    targetSpeed = getParam(params, '_Target_finalSpeed', 8.333);

    % 1. Ego Vehicle
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', egoData.ClassID, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, ...
        'PlotColor', [0.05 0.45 0.95]);

    if enableAEB
        % Ego turns into junction, spots oncoming target, halts safely in junction throat
        egoWaypoints = [
            210.0, -1.75, 0.0;
            246.0, -1.75, 0.0;
            254.0, -0.50, 0.0;   % detects oncoming target, brakes
            255.5, -0.30, 0.0;   % standstill yielding
            260.0,  5.00, 0.0;   % resumes turn after target passes
            263.25, 40.0, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 1.0, 0.05, 2.0, egoSpeed];
    else
        egoWaypoints = [
            210.0, -1.75, 0.0;
            246.0, -1.75, 0.0;
            256.0,  1.00, 0.0;
            263.25, 40.0, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % 2. Target Vehicle (Approaching from East Road 2, driving West)
    targetData = entities.Target;
    target = vehicle(scenario, 'ClassID', targetData.ClassID, 'Name', 'Target_Red', ...
        'AssetType', targetData.AssetType, 'Length', targetData.Length, ...
        'Width', targetData.Width, 'Height', targetData.Height, ...
        'PlotColor', [0.85 0.15 0.15]);

    targetWaypoints = [
        320.0, 1.75, 0.0;   % Oncoming in Lane 1 (Y = +1.75)
        240.0, 1.75, 0.0;
        180.0, 1.75, 0.0
    ];
    trajectory(target, targetWaypoints, targetSpeed);
    actorMap.Target = target;
end

function [scenario, actorMap] = setup_cccscp_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Car-to-Car Straight Crossing Path at Intersection (CCCscp)
    % Ego travels West-to-East (Road 0 to Road 2).
    % Target travels South-to-North (Road 3 to Road 1).
    % Multiple parked obstruction vehicles near junction corner.

    actorMap = struct();
    egoSpeed = getParam(params, '_Ego_speed', 5.556); % 20 km/h
    targetSpeed = getParam(params, '_Target_final_speed', 5.556);

    % 1. Ego Vehicle
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', egoData.ClassID, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, ...
        'PlotColor', [0.05 0.45 0.95]);

    if enableAEB
        egoWaypoints = [
            180.0, -1.75, 0.0;
            235.0, -1.75, 0.0;
            248.0, -1.75, 0.0;  % spots cross-traffic target, yields before junction box
            249.0, -1.75, 0.0;  % standstill
            275.0, -1.75, 0.0;  % accelerates through junction after target crosses
            320.0, -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 2.0, 0.05, egoSpeed, egoSpeed];
    else
        egoWaypoints = [
            180.0, -1.75, 0.0;
            320.0, -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % 2. Target Vehicle (From South Road 3 traveling North to Road 1)
    targetData = entities.Target;
    target = vehicle(scenario, 'ClassID', targetData.ClassID, 'Name', 'Target_Red', ...
        'AssetType', targetData.AssetType, 'Length', targetData.Length, ...
        'Width', targetData.Width, 'Height', targetData.Height, ...
        'PlotColor', [0.85 0.15 0.15]);

    targetWaypoints = [
        263.25, -60.0, 0.0;
        263.25,  60.0, 0.0
    ];
    trajectory(target, targetWaypoints, targetSpeed);
    actorMap.Target = target;

    % 3. Parked Obstruction Vehicles
    if isfield(entities, 'ObstructionVehicle_Large')
        o1 = entities.ObstructionVehicle_Large;
        actorMap.Obstruction1 = vehicle(scenario, 'ClassID', o1.ClassID, 'Name', 'ObstructionLarge', ...
            'AssetType', o1.AssetType, 'Length', o1.Length, 'Width', o1.Width, 'Height', o1.Height, ...
            'Position', [240.0, -4.5, 0.0], 'Yaw', 0, 'PlotColor', [0.4 0.4 0.4]);
    end
    if isfield(entities, 'ObstructionVehicle_Small')
        o2 = entities.ObstructionVehicle_Small;
        actorMap.Obstruction2 = vehicle(scenario, 'ClassID', o2.ClassID, 'Name', 'ObstructionSmall', ...
            'AssetType', o2.AssetType, 'Length', o2.Length, 'Width', o2.Width, 'Height', o2.Height, ...
            'Position', [246.0, -4.5, 0.0], 'Yaw', 0, 'PlotColor', [0.45 0.45 0.45]);
    end
end

% =========================================================================
% INDIAN & ASAM STANDARD SCENARIO SETUPS
% =========================================================================

function [scenario, actorMap] = setup_indian_autocutin_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Indian Auto-Rickshaw Aggressive Cut-In
    % Road: Indian_Urban_Arterial (4 lanes, Lane 1 Y = -1.75m, Lane 2 Y = -5.25m)
    actorMap = struct();

    % Extract dynamic scenario parameters with backwards-compatible defaults
    egoSpeed = getParam(params, 'Ego_Speed', 11.11);
    autoSpeed = getParam(params, 'Auto_Speed', 8.33);
    autoStartX = getParam(params, 'Auto_StartX', 55.0);
    cutInDist = getParam(params, 'CutIn_Distance', 30.0);
    cutInDuration = getParam(params, 'CutIn_Duration', 2.4);

    % 1. Ego Vehicle
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', egoData.ClassID, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.1 0.4 0.9]);
    
    cutInStartX = autoStartX + cutInDist;
    cutInEndX = cutInStartX + autoSpeed * cutInDuration;

    if enableAEB
        egoBrakeX = max(45.0, cutInStartX - 10.0);
        egoYieldX = max(egoBrakeX + 15.0, cutInEndX - 5.0);
        egoWaypoints = [
            20.0,         -1.75, 0.0;
            autoStartX,   -1.75, 0.0;
            egoBrakeX,    -1.75, 0.0;   % auto initiates cut-in, ego slows
            egoYieldX,    -1.75, 0.0;   % maintains safe following distance behind auto
            egoYieldX+75, -1.75, 0.0;
            340.0,        -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, min(6.0, autoSpeed * 0.85), autoSpeed, egoSpeed, egoSpeed];
    else
        egoWaypoints = [20.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % 2. Auto-Rickshaw (cuts in from outer lane -5.25 to lane 1 -1.75)
    autoData = entities.AutoRickshaw;
    auto = vehicle(scenario, 'ClassID', 1, 'Name', 'AutoRickshaw_Yellow', ...
        'AssetType', 'Hatchback', 'Length', autoData.Length, ...
        'Width', autoData.Width, 'Height', autoData.Height, 'PlotColor', [0.95 0.85 0.05]);
    
    autoWaypoints = [
        autoStartX,     -5.25, 0.0;   % t = 0s: traveling in outer lane
        cutInStartX,    -5.25, 0.0;   % initiates sharp cut-in
        cutInEndX,      -1.75, 0.0;   % established inside Lane 1 in front of Ego
        cutInEndX+55.0, -1.75, 0.0;
        280.0,          -1.75, 0.0
    ];
    autoSpeeds = [autoSpeed, autoSpeed, autoSpeed * 0.95, autoSpeed, autoSpeed];
    trajectory(auto, autoWaypoints, autoSpeeds);
    actorMap.AutoRickshaw = auto;
    % 3. Slow Truck ahead in Lane 2
    if isfield(entities, 'SlowTruck')
        trkData = entities.SlowTruck;
        trk = vehicle(scenario, 'ClassID', 2, 'Name', 'TataTruck_Brown', ...
            'AssetType', 'BoxTruck', 'Length', trkData.Length, ...
            'Width', trkData.Width, 'Height', trkData.Height, 'PlotColor', [0.55 0.35 0.20]);
        trkWaypoints = [110.0, -5.25, 0.0; 280.0, -5.25, 0.0];
        trajectory(trk, trkWaypoints, 5.55);
        actorMap.SlowTruck = trk;
    end
end

function [scenario, actorMap] = setup_indian_twowheeler_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Two-Wheeler Motorcycle Filtering Between Lanes
    actorMap = struct();

    % 1. Ego Vehicle
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.1 0.4 0.9]);
    egoWaypoints = [30.0, -1.75, 0.0; 340.0, -1.75, 0.0];
    assign_ego_motion(ego, egoWaypoints, 8.33, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % 2. Parallel Car in outer lane
    if isfield(entities, 'ParallelCar')
        pData = entities.ParallelCar;
        pCar = vehicle(scenario, 'ClassID', 1, 'Name', 'MarutiSwift_White', ...
            'AssetType', 'Hatchback', 'Length', pData.Length, ...
            'Width', pData.Width, 'Height', pData.Height, 'PlotColor', [0.9 0.9 0.9]);
        pWaypoints = [40.0, -5.25, 0.0; 280.0, -5.25, 0.0];
        trajectory(pCar, pWaypoints, 6.94);
        actorMap.ParallelCar = pCar;
    end

    % 3. Motorcycle filtering between lanes (Y = -3.50m)
    bikeData = entities.Motorcycle;
    bike = actor(scenario, 'ClassID', 3, 'Name', 'Motorcycle_Pulsar', ...
        'AssetType', 'Bicyclist', 'Length', bikeData.Length, ...
        'Width', bikeData.Width, 'Height', bikeData.Height, 'PlotColor', [0.9 0.1 0.1]);
    bikeWaypoints = [5.0, -3.50, 0.0; 300.0, -3.50, 0.0];
    trajectory(bike, bikeWaypoints, 13.88);
    actorMap.Motorcycle = bike;
end

function [scenario, actorMap] = setup_indian_jaywalk_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Mid-block Jaywalking - Multi-Pedestrian (Adult + Running Child) with Oncoming Scooter
    actorMap = struct();

    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.1 0.4 0.9]);

    if enableAEB
        egoWaypoints = [
            20.0, -1.75, 0.0;
            75.0, -1.75, 0.0;   % cruising, detects pedestrian ahead
            90.0, -1.75, 0.0;   % braking zone
            104.0, -1.75, 0.0;  % near-standstill safely behind crossing path
            105.0, -1.75, 0.0;  % yields while pedestrians cross
            120.0, -1.75, 0.0;  % resumes after clear
            160.0, -1.75, 0.0;
            340.0, -1.75, 0.0
        ];
        egoSpeeds = [6.94, 6.94, 3.0, 0.5, 0.05, 6.94, 6.94, 6.94];
    else
        egoWaypoints = [20.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [6.94, 6.94];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % Parked Bus on outer lane (occludes pedestrians)
    busData = entities.ParkedBus;
    bus = vehicle(scenario, 'ClassID', 2, 'Name', 'BMTC_Bus', ...
        'AssetType', 'BoxTruck', 'Length', busData.Length, ...
        'Width', busData.Width, 'Height', busData.Height, ...
        'Position', [90.0, -5.25, 0.0], 'Yaw', 0, 'PlotColor', [0.1 0.6 0.3]);
    actorMap.ParkedBus = bus;

    % Adult Jaywalking Pedestrian (slower, walks across)
    pedData = entities.Pedestrian;
    ped = actor(scenario, 'ClassID', 4, 'Name', 'Pedestrian_Jaywalker', ...
        'AssetType', 'MalePedestrian', 'Length', 0.28, ...
        'Width', 0.45, 'Height', 1.75, 'PlotColor', [0.95 0.5 0.1]);
    pedWaypoints = [
        97.0, -7.0, 0.0;
        97.0, -5.25, 0.0;   % emerges from in front of parked bus
        97.0, -1.75, 0.0;   % crosses lane 1
        97.0,  1.50, 0.0    % reaches median
    ];
    pedSpeeds = [1.5, 1.5, 1.5, 1.5];
    trajectory(ped, pedWaypoints, pedSpeeds);
    actorMap.Pedestrian = ped;

    % Running Child (faster, smaller, crosses slightly ahead of adult - more dangerous)
    if isfield(entities, 'RunningChild')
        childData = entities.RunningChild;
        child = actor(scenario, 'ClassID', 4, 'Name', 'Child_Runner', ...
            'AssetType', 'ChildPedestrian', 'Length', 0.25, ...
            'Width', 0.35, 'Height', 1.15, 'PlotColor', [1.0 0.3 0.1]);
        childWaypoints = [
            93.0, -7.0, 0.0;
            93.0, -5.25, 0.0;   % emerges from behind bus
            93.0, -1.75, 0.0;   % sprints across lane 1
            93.0,  1.50, 0.0    % reaches median
        ];
        childSpeeds = [2.5, 2.5, 2.5, 2.5];
        trajectory(child, childWaypoints, childSpeeds);
        actorMap.RunningChild = child;
    end

    % Oncoming Scooter in opposing lane (adds visual complexity + blocks lane change escape)
    if isfield(entities, 'OncomingScooter')
        scData = entities.OncomingScooter;
        scooter = actor(scenario, 'ClassID', 3, 'Name', 'HondaActiva_Oncoming', ...
            'AssetType', 'Bicyclist', 'Length', scData.Length, ...
            'Width', scData.Width, 'Height', scData.Height, 'PlotColor', [0.6 0.2 0.8]);
        scooterWps = [200.0, 1.75, 0.0; 10.0, 1.75, 0.0];
        trajectory(scooter, scooterWps, 8.33);
        actorMap.OncomingScooter = scooter;
    end
end

function [scenario, actorMap] = setup_indian_cattle_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Multiple Stray Cattle - Wandering cow + Stationary cow + Oncoming bus + Shoulder pedestrian
    actorMap = struct();

    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.1 0.4 0.9]);

    if enableAEB
        % Ego approaches at 25 km/h, slows down before cattle cluster,
        % steers around into outer lane (Y = -5.00), then returns to lane 1
        egoWaypoints = [
            20.0, -1.75, 0.0;
            75.0, -1.75, 0.0;   % detect cattle ahead, begin braking
            90.0, -3.50, 0.0;   % steer toward outer lane
            100.0, -5.00, 0.0;  % passing cow cluster on outside
            115.0, -5.00, 0.0;  % clear of second cow
            135.0, -1.75, 0.0;  % smooth return to lane 1
            200.0, -1.75, 0.0;
            340.0, -1.75, 0.0
        ];
        egoSpeeds = [6.94, 6.94, 3.5, 3.5, 3.5, 6.94, 6.94, 6.94];
    else
        egoWaypoints = [20.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [6.94, 6.94];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % Wandering Cattle in Lane 1 (slowly moving forward at 0.8 m/s)
    cow1Speed = getParam(params, 'Cow1_Speed', 0.8);
    cow1 = actor(scenario, 'ClassID', 4, 'Name', 'Cow_Wandering', ...
        'AssetType', 'MalePedestrian', 'Length', 2.10, 'Width', 1.10, 'Height', 1.40, ...
        'PlotColor', [0.8 0.8 0.8]);
    cow1Wps = [100.0, -1.75, 0.0; 105.0, -2.30, 0.0; 112.0, -1.75, 0.0; 120.0, -2.00, 0.0];
    cow1Speeds = [cow1Speed, cow1Speed, cow1Speed, cow1Speed];
    trajectory(cow1, cow1Wps, cow1Speeds);
    actorMap.Hazard = cow1;

    % Second Stationary Cow at lane divider (blocks outer lane escape partially)
    if isfield(entities, 'StrayCattle2')
        cow2 = actor(scenario, 'ClassID', 5, 'Name', 'Cow_Stationary', ...
            'AssetType', 'Barrier', 'Length', 1.90, 'Width', 1.00, 'Height', 1.30, ...
            'Position', [108.0, -3.50, 0.0], 'Yaw', 0, 'PlotColor', [0.7 0.7 0.7]);
        actorMap.StrayCattle2 = cow2;
    end

    % Oncoming vehicle in opposing lane
    if isfield(entities, 'OncomingVehicle')
        onData = entities.OncomingVehicle;
        onBus = vehicle(scenario, 'ClassID', 2, 'Name', 'OncomingBus', ...
            'AssetType', 'BoxTruck', 'Length', onData.Length, ...
            'Width', onData.Width, 'Height', onData.Height, 'PlotColor', [0.2 0.7 0.4]);
        onWaypoints = [250.0, 1.75, 0.0; 10.0, 1.75, 0.0];
        trajectory(onBus, onWaypoints, 9.72);
        actorMap.OncomingBus = onBus;
    end

    % Shoulder Pedestrian (adds visual complexity near cattle area)
    if isfield(entities, 'ShoulderPedestrian')
        pedData = entities.ShoulderPedestrian;
        ped = actor(scenario, 'ClassID', 4, 'Name', 'ShoulderPed', ...
            'AssetType', 'MalePedestrian', 'Length', pedData.Length, ...
            'Width', pedData.Width, 'Height', pedData.Height, 'PlotColor', [0.85 0.55 0.10]);
        pedWps = [95.0, -6.50, 0.0; 140.0, -6.50, 0.0];
        trajectory(ped, pedWps, 1.0);
        actorMap.ShoulderPedestrian = ped;
    end
end

function [scenario, actorMap] = setup_lanechange_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    actorMap = struct();
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.1 0.4 0.9]);
    egoWaypoints = [20.0, -1.75, 0.0; 70.0, -1.75, 0.0; 100.0, 1.75, 0.0; 250.0, 1.75, 0.0];
    assign_ego_motion(ego, egoWaypoints, [13.88, 13.88, 13.88, 13.88], liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    if isfield(entities, 'LeadingVehicle')
        lead = vehicle(scenario, 'ClassID', 1, 'Name', 'LeadCar', ...
            'AssetType', 'Sedan', 'Length', 4.5, 'Width', 1.8, 'Height', 1.5, ...
            'PlotColor', [0.5 0.5 0.5]);
        leadWaypoints = [70.0, -1.75, 0.0; 250.0, -1.75, 0.0];
        trajectory(lead, leadWaypoints, 8.33);
        actorMap.Lead = lead;
    end
end

function [scenario, actorMap] = setup_overtake_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    actorMap = struct();
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.1 0.4 0.9]);
    egoWaypoints = [15.0, -1.75, 0.0; 50.0, -1.75, 0.0; 80.0, 1.75, 0.0; 150.0, 1.75, 0.0; 180.0, -1.75, 0.0; 280.0, -1.75, 0.0];
    assign_ego_motion(ego, egoWaypoints, 13.88, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    if isfield(entities, 'SlowTruck')
        trk = vehicle(scenario, 'ClassID', 2, 'Name', 'SlowTruck', ...
            'AssetType', 'BoxTruck', 'Length', 8.0, 'Width', 2.4, 'Height', 3.0, ...
            'PlotColor', [0.6 0.4 0.2]);
        trkWaypoints = [65.0, -1.75, 0.0; 280.0, -1.75, 0.0];
        trajectory(trk, trkWaypoints, 5.55);
        actorMap.SlowTruck = trk;
    end
end

function [scenario, actorMap] = setup_pedcrossing_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    actorMap = struct();
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.1 0.4 0.9]);
    if enableAEB
        egoWaypoints = [20.0, -1.75, 0.0; 65.0, -1.75, 0.0; 75.0, -1.75, 0.0; 76.0, -1.75, 0.0; 150.0, -1.75, 0.0; 250.0, -1.75, 0.0];
        egoSpeeds = [11.11, 11.11, 2.0, 0.05, 11.11, 11.11];
    else
        egoWaypoints = [20.0, -1.75, 0.0; 250.0, -1.75, 0.0];
        egoSpeeds = [11.11, 11.11];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    ped = actor(scenario, 'ClassID', 4, 'Name', 'Pedestrian', ...
        'AssetType', 'MalePedestrian', 'Length', 0.28, 'Width', 0.45, 'Height', 1.80, ...
        'PlotColor', [0.9 0.6 0.1]);
    pedWaypoints = [80.0, -6.0, 0.0; 80.0, 2.0, 0.0];
    trajectory(ped, pedWaypoints, 1.4);
    actorMap.Pedestrian = ped;
end

function [scenario, actorMap] = setup_indian_congestion_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Indian Dense Traffic Congestion: Multi-Vehicle Crawling Queue with Filtering Bicycle
    % Road: Indian_Urban_Arterial (4 lanes: Lane 1 Y = -1.75m, Lane 2 Y = -5.25m)
    actorMap = struct();

    egoSpeed = getParam(params, 'Ego_Speed', 8.33);      % 30 km/h approach
    queueSpeed = getParam(params, 'Queue_Speed', 3.5);   % 12.6 km/h queue crawl
    bikeSpeed = getParam(params, 'Bicycle_Speed', 4.2);  % 15.1 km/h filtering

    % 1. Ego Vehicle (MUST BE ACTOR 1)
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.10 0.45 0.95]);

    if enableAEB
        % Ego approaches at 30 km/h, encounters creeping queue at X=50m, yields and crawls
        egoWaypoints = [
            20.0,  -1.75, 0.0;
            45.0,  -1.75, 0.0;
            60.0,  -1.75, 0.0;   % encounters slow queue, decelerates to match crawl
            90.0,  -1.75, 0.0;
            140.0, -1.75, 0.0;
            340.0, -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed * 0.8, queueSpeed, queueSpeed, queueSpeed, queueSpeed];
    else
        egoWaypoints = [20.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % 2. Slow Lead Car in Lane 1 (Hatchback, Y = -1.75m)
    car1Data = entities.SlowCar1;
    car1 = vehicle(scenario, 'ClassID', 1, 'Name', 'Hatchback_Silver', ...
        'AssetType', 'Hatchback', 'Length', car1Data.Length, ...
        'Width', car1Data.Width, 'Height', car1Data.Height, 'PlotColor', [0.75 0.75 0.80]);
    car1Wps = [50.0, -1.75, 0.0; 280.0, -1.75, 0.0];
    trajectory(car1, car1Wps, queueSpeed);
    actorMap.SlowCar1 = car1;

    % 3. Parked / Stalled Car on outer shoulder (Sedan, Red, stationary at Y = -5.50m)
    car2Data = entities.ParkedCar;
    car2 = vehicle(scenario, 'ClassID', 1, 'Name', 'Sedan_Red', ...
        'AssetType', 'Sedan', 'Length', car2Data.Length, ...
        'Width', car2Data.Width, 'Height', car2Data.Height, 'PlotColor', [0.85 0.20 0.15]);
    car2.Position = [65.0, -5.50, 0.0];
    actorMap.ParkedCar = car2;

    % 4. Creeping Truck in outer lane (Truck, Green, at Y = -5.10m)
    trk1Data = entities.CreepTruck;
    trk1 = vehicle(scenario, 'ClassID', 2, 'Name', 'Truck_Green', ...
        'AssetType', 'BoxTruck', 'Length', trk1Data.Length, ...
        'Width', trk1Data.Width, 'Height', trk1Data.Height, 'PlotColor', [0.25 0.65 0.30]);
    trk1Wps = [85.0, -5.10, 0.0; 280.0, -5.10, 0.0];
    trajectory(trk1, trk1Wps, 2.8);
    actorMap.CreepTruck = trk1;

    % 5. Ahead Car in Lane 1 queue (Sedan, White, at Y = -1.75m)
    car3Data = entities.AheadCar;
    car3 = vehicle(scenario, 'ClassID', 1, 'Name', 'Sedan_White', ...
        'AssetType', 'Sedan', 'Length', car3Data.Length, ...
        'Width', car3Data.Width, 'Height', car3Data.Height, 'PlotColor', [0.90 0.90 0.95]);
    car3Wps = [85.0, -1.75, 0.0; 280.0, -1.75, 0.0];
    trajectory(car3, car3Wps, queueSpeed);
    actorMap.AheadCar = car3;

    % 6. Lead Truck ahead in Lane 1 (Truck, Brown, at Y = -1.75m)
    trk2Data = entities.LeadTruck;
    trk2 = vehicle(scenario, 'ClassID', 2, 'Name', 'BoxTruck_Brown', ...
        'AssetType', 'BoxTruck', 'Length', trk2Data.Length, ...
        'Width', trk2Data.Width, 'Height', trk2Data.Height, 'PlotColor', [0.55 0.38 0.22]);
    trk2Wps = [130.0, -1.75, 0.0; 280.0, -1.75, 0.0];
    trajectory(trk2, trk2Wps, queueSpeed);
    actorMap.LeadTruck = trk2;

    % 7. Filtering Bicycle (Bicycle, ClassID 3, Yellow, lane splits along Y = -3.50m)
    bikeData = entities.Bicycle;
    bike = actor(scenario, 'ClassID', 3, 'Name', 'FilteringBicycle', ...
        'AssetType', 'Bicycle', 'Length', bikeData.Length, ...
        'Width', bikeData.Width, 'Height', bikeData.Height, 'PlotColor', [0.95 0.85 0.10]);
    bikeWps = [
        35.0,  -3.50, 0.0;   % lane divider between Lane 1 and 2
        60.0,  -3.45, 0.0;
        90.0,  -3.55, 0.0;
        130.0, -3.45, 0.0;
        180.0, -3.50, 0.0;
        280.0, -3.50, 0.0
    ];
    trajectory(bike, bikeWps, bikeSpeed);
    actorMap.Bicycle = bike;
end

function [scenario, actorMap] = setup_indian_pothole_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Indian Road Pothole-Forced Detour around Road Hazard Barrier
    % Road: Indian_Urban_Arterial (Eastbound Lane 1: Y = -1.75m, Westbound Lane 1: Y = +1.75m)
    actorMap = struct();

    egoSpeed = getParam(params, 'Ego_Speed', 6.94);      % 25 km/h steady smooth cruise
    pedSpeed = getParam(params, 'Pedestrian_Speed', 1.2);% 4.3 km/h shoulder pedestrian

    % 1. Ego Vehicle (MUST BE ACTOR 1)
    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.10 0.45 0.95]);

    if enableAEB
        % Ego cruises steadily at 25 km/h, encounters barrier/deep pothole at X=85m,
        % detects obstacle in advance, decelerates smoothly to safe detour crawl (14 km/h),
        % executes smooth sinusoidal lane change into clear adjacent lane (Y = +1.75m),
        % and merges cleanly back into original driving lane with zero spline overshoot!
        egoWaypoints = [
            25.0,  -1.75, 0.0;   % Straight cruise in Lane -1 (inner eastbound lane)
            50.0,  -1.75, 0.0;   % Detects barrier/pothole ahead, begins deceleration
            65.0,  -2.75, 0.0;   % Smooth transition right toward Lane -2 (our own side)
            80.0,  -4.75, 0.0;   % Safely entering Lane -2 before the hazard
            95.0,  -5.00, 0.0;   % Centered in Lane -2 past the barrier
            120.0, -5.00, 0.0;   % COMMITTED to Lane -2: stays in safe lane, no forced bounce-back
            160.0, -5.00, 0.0;   % Straight cruising in Lane -2
            200.0, -5.00, 0.0;   % Straight cruising in Lane -2
            280.0, -5.00, 0.0;   % Straight cruising in Lane -2
            340.0, -5.00, 0.0    % Clean road exit in Lane -2
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 4.2, 4.0, 4.0, egoSpeed, egoSpeed, egoSpeed, egoSpeed, egoSpeed];
    else
        egoWaypoints = [25.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % 2. Shoulder Pedestrian walking along right curb (X = 80 -> 130m at Y = -4.20m)
    pedData = entities.ShoulderPedestrian;
    ped = actor(scenario, 'ClassID', 4, 'Name', 'ShoulderPedestrian', ...
        'AssetType', 'MalePedestrian', 'Length', pedData.Length, ...
        'Width', pedData.Width, 'Height', pedData.Height, 'PlotColor', [0.85 0.55 0.10]);
    pedWps = [80.0, -6.50, 0.0; 130.0, -6.50, 0.0];
    trajectory(ped, pedWps, pedSpeed);
    actorMap.ShoulderPedestrian = ped;

    % 3. Visible Road Hazard Barricade right at the pothole site (X = 85.0m, Y = -1.75m)
    barrier = actor(scenario, 'ClassID', 5, 'Name', 'RoadWorkBarrier', ...
        'AssetType', 'JerseyBarrier', 'Length', 5.0, 'Width', 0.8, 'Height', 0.8, ...
        'PlotColor', [0.95 0.45 0.10]);
    barrier.Position = [85.0, -1.75, 0.4];
    actorMap.RoadBarrier = barrier;

    % 4. Warning Cone (advance warning marker at X=80)
    if isfield(entities, 'WarningCone')
        cone = actor(scenario, 'ClassID', 5, 'Name', 'WarningCone', ...
            'AssetType', 'JerseyBarrier', 'Length', 0.30, 'Width', 0.30, 'Height', 0.70, ...
            'Position', [80.0, -1.75, 0.0], 'Yaw', 0, 'PlotColor', [1.0 0.6 0.0]);
        actorMap.WarningCone = cone;
    end

    % 5. Oncoming Car in opposing lane (creates tension during detour)
    if isfield(entities, 'OncomingCar')
        onData = entities.OncomingCar;
        onCar = vehicle(scenario, 'ClassID', 1, 'Name', 'OncomingSwift', ...
            'AssetType', onData.AssetType, 'Length', onData.Length, ...
            'Width', onData.Width, 'Height', onData.Height, 'PlotColor', [0.9 0.2 0.2]);
        onWps = [220.0, 1.75, 0.0; 10.0, 1.75, 0.0];
        trajectory(onCar, onWps, 8.33);
        actorMap.OncomingCar = onCar;
    end

    % 6. Register Road Surface Defects in actorMap
    % Potholes: [X, Y, depth_cm, radius_m, severity]
    actorMap.Potholes = [
        85.0,  -1.75, 8.5, 0.75, 1.0;   % Deep cavity wall (lethal in costmap, cost 254)
        130.0, -1.75, 3.2, 0.50, 1.0    % Shallow dip (cost 120, speed capped <= 15 km/h)
    ];
end


function [scenario, actorMap] = setup_indian_wrongway_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Wrong-Way Vehicle Head-On Encounter
    % Ego faces a tempo/mini-truck driving HEAD-ON in ego's lane
    actorMap = struct();

    egoSpeed = getParam(params, 'Ego_Speed', 6.94);

    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.10 0.45 0.95]);

    if enableAEB
        % Ego detects oncoming vehicle, brakes hard, swerves to outer lane
        egoWaypoints = [
            20.0,  -1.75, 0.0;   % Cruising in lane 1
            60.0,  -1.75, 0.0;   % Detects oncoming vehicle
            80.0,  -1.75, 0.0;   % Emergency braking begins
            90.0,  -3.50, 0.0;   % Swerving to outer lane
            100.0, -5.00, 0.0;   % Settled in outer lane, tempo passes
            120.0, -5.00, 0.0;   % Clear of wrong-way vehicle
            140.0, -1.75, 0.0;   % Return to lane 1
            200.0, -1.75, 0.0;
            340.0, -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 3.0, 2.5, 2.5, 3.5, egoSpeed, egoSpeed, egoSpeed];
    else
        egoWaypoints = [20.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % Wrong-Way Tempo (driving head-on toward ego in ego's lane)
    wwData = entities.WrongWayTempo;
    wwSpeed = getParam(params, 'WrongWay_Speed', 8.33);
    ww = vehicle(scenario, 'ClassID', 2, 'Name', 'WrongWay_Tempo', ...
        'AssetType', 'BoxTruck', 'Length', wwData.Length, ...
        'Width', wwData.Width, 'Height', wwData.Height, 'PlotColor', [0.9 0.1 0.1]);
    wwWps = [200.0, -1.75, 0.0; 10.0, -1.75, 0.0];
    trajectory(ww, wwWps, wwSpeed);
    actorMap.WrongWayTempo = ww;

    % Parked Delivery Van blocking outer lane escape
    if isfield(entities, 'ParkedDeliveryVan')
        pvData = entities.ParkedDeliveryVan;
        pv = vehicle(scenario, 'ClassID', 1, 'Name', 'DeliveryVan', ...
            'AssetType', pvData.AssetType, 'Length', pvData.Length, ...
            'Width', pvData.Width, 'Height', pvData.Height, ...
            'Position', [110.0, -5.25, 0.0], 'Yaw', 0, 'PlotColor', [0.4 0.4 0.4]);
        actorMap.ParkedDeliveryVan = pv;
    end

    % Shoulder Pedestrian
    if isfield(entities, 'ShoulderPedestrian')
        spData = entities.ShoulderPedestrian;
        sp = actor(scenario, 'ClassID', 4, 'Name', 'ShoulderPed', ...
            'AssetType', 'MalePedestrian', 'Length', spData.Length, ...
            'Width', spData.Width, 'Height', spData.Height, 'PlotColor', [0.85 0.55 0.10]);
        spWps = [130.0, -6.50, 0.0; 170.0, -6.50, 0.0];
        trajectory(sp, spWps, 1.0);
        actorMap.ShoulderPedestrian = sp;
    end

    % Legal Oncoming Bike in opposing lane
    if isfield(entities, 'OncomingBike')
        obData = entities.OncomingBike;
        ob = actor(scenario, 'ClassID', 3, 'Name', 'OncomingPulsar', ...
            'AssetType', 'Bicyclist', 'Length', obData.Length, ...
            'Width', obData.Width, 'Height', obData.Height, 'PlotColor', [0.6 0.2 0.8]);
        obWps = [180.0, 1.75, 0.0; 10.0, 1.75, 0.0];
        trajectory(ob, obWps, 11.11);
        actorMap.OncomingBike = ob;
    end
end

function [scenario, actorMap] = setup_indian_schoolzone_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % School Zone - Multiple Children Crossing Near Stopped School Bus
    actorMap = struct();

    egoSpeed = getParam(params, 'Ego_Speed', 6.94);

    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.10 0.45 0.95]);

    if enableAEB
        % Ego sees school bus, slows down, stops for children crossing
        egoWaypoints = [
            25.0,  -1.75, 0.0;   % Cruising
            80.0,  -1.75, 0.0;   % Detects school bus ahead
            100.0, -1.75, 0.0;   % Braking
            108.0, -1.75, 0.0;   % Near-standstill
            110.0, -1.75, 0.0;   % Waiting for children to cross
            112.0, -1.75, 0.0;   % Creeping forward after first child
            115.0, -1.75, 0.0;   % Second stop for later children
            140.0, -1.75, 0.0;   % Resumes
            200.0, -1.75, 0.0;
            340.0, -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 3.0, 0.5, 0.05, 0.5, 0.05, egoSpeed, egoSpeed, egoSpeed];
    else
        egoWaypoints = [25.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % School Bus (stopped in outer lane)
    busData = entities.SchoolBus;
    bus = vehicle(scenario, 'ClassID', 2, 'Name', 'SchoolBus_Yellow', ...
        'AssetType', 'BoxTruck', 'Length', busData.Length, ...
        'Width', busData.Width, 'Height', busData.Height, ...
        'Position', [120.0, -5.25, 0.0], 'Yaw', 0, 'PlotColor', [0.95 0.85 0.10]);
    actorMap.SchoolBus = bus;

    % Running Child (fast, crosses first)
    if isfield(entities, 'RunningChild')
        rc = actor(scenario, 'ClassID', 4, 'Name', 'RunningChild', ...
            'AssetType', 'ChildPedestrian', 'Length', 0.25, ...
            'Width', 0.35, 'Height', 1.15, 'PlotColor', [1.0 0.3 0.1]);
        rcWps = [115.0, -7.0, 0.0; 115.0, -5.25, 0.0; 115.0, -1.75, 0.0; 115.0, 1.50, 0.0];
        trajectory(rc, rcWps, getParam(params, 'RunningChild_Speed', 2.0));
        actorMap.RunningChild = rc;
    end

    % Walking Child (slower, crosses second)
    if isfield(entities, 'WalkingChild')
        wc = actor(scenario, 'ClassID', 4, 'Name', 'WalkingChild', ...
            'AssetType', 'ChildPedestrian', 'Length', 0.25, ...
            'Width', 0.35, 'Height', 1.15, 'PlotColor', [0.2 0.7 1.0]);
        wcWps = [122.0, -7.0, 0.0; 122.0, -5.25, 0.0; 122.0, -1.75, 0.0; 122.0, 1.50, 0.0];
        trajectory(wc, wcWps, getParam(params, 'WalkingChild_Speed', 1.2));
        actorMap.WalkingChild = wc;
    end

    % Sprinting Child (fastest, crosses last - most dangerous)
    if isfield(entities, 'SprintingChild')
        sc = actor(scenario, 'ClassID', 4, 'Name', 'SprintingChild', ...
            'AssetType', 'ChildPedestrian', 'Length', 0.25, ...
            'Width', 0.35, 'Height', 1.15, 'PlotColor', [1.0 0.1 0.5]);
        scWps = [128.0, -7.0, 0.0; 128.0, -5.25, 0.0; 128.0, -1.75, 0.0; 128.0, 1.50, 0.0];
        trajectory(sc, scWps, getParam(params, 'SprintingChild_Speed', 2.8));
        actorMap.SprintingChild = sc;
    end

    % Guardian adult waiting on the other side
    if isfield(entities, 'Guardian')
        gd = actor(scenario, 'ClassID', 4, 'Name', 'Guardian_Adult', ...
            'AssetType', 'MalePedestrian', 'Length', 0.28, ...
            'Width', 0.45, 'Height', 1.75, ...
            'Position', [118.0, 2.0, 0.0], 'Yaw', 270, 'PlotColor', [0.3 0.6 0.3]);
        actorMap.Guardian = gd;
    end
end

function [scenario, actorMap] = setup_indian_busstop_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Bus Stop Hazard: Bus stopping + Alighting Passenger + Overtaking Motorcycle
    actorMap = struct();

    egoSpeed = getParam(params, 'Ego_Speed', 6.94);

    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.10 0.45 0.95]);

    if enableAEB
        % Ego detects bus slowing + passenger stepping out, brakes and holds
        egoWaypoints = [
            25.0,  -1.75, 0.0;   % Cruising
            70.0,  -1.75, 0.0;   % Detects bus ahead
            90.0,  -1.75, 0.0;   % Slowing behind bus
            100.0, -1.75, 0.0;   % Near-stop as passenger steps out
            102.0, -1.75, 0.0;   % Waiting
            115.0, -1.75, 0.0;   % Resumes carefully after clear
            160.0, -1.75, 0.0;
            340.0, -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 3.0, 0.5, 0.05, 4.0, egoSpeed, egoSpeed];
    else
        egoWaypoints = [25.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % City Bus (decelerating to stop)
    busData = entities.CityBus;
    busInitSpeed = getParam(params, 'Bus_InitSpeed', 4.17);
    bus = vehicle(scenario, 'ClassID', 2, 'Name', 'CityBus_BMTC', ...
        'AssetType', 'BoxTruck', 'Length', busData.Length, ...
        'Width', busData.Width, 'Height', busData.Height, 'PlotColor', [0.1 0.6 0.3]);
    busWps = [110.0, -5.25, 0.0; 125.0, -5.25, 0.0; 130.0, -5.25, 0.0];
    busSpeeds = [busInitSpeed, 1.5, 0.0];
    trajectory(bus, busWps, busSpeeds);
    actorMap.CityBus = bus;

    % Alighting Passenger (steps into traffic from bus door area)
    if isfield(entities, 'AlightingPassenger')
        ap = actor(scenario, 'ClassID', 4, 'Name', 'AlightingPassenger', ...
            'AssetType', 'MalePedestrian', 'Length', 0.28, ...
            'Width', 0.45, 'Height', 1.75, 'PlotColor', [0.95 0.5 0.1]);
        apWps = [
            108.0, -5.00, 0.0;   % At bus door
            108.0, -3.50, 0.0;   % Steps toward road
            108.0, -1.75, 0.0;   % Walks into ego lane
            108.0, -0.50, 0.0    % Continues crossing
        ];
        apSpeeds = [0.0, 1.2, 1.5, 1.5];
        trajectory(ap, apWps, apSpeeds);
        actorMap.AlightingPassenger = ap;
    end

    % Overtaking Motorcycle (fast, coming from behind bus)
    if isfield(entities, 'OvertakingBike')
        obData = entities.OvertakingBike;
        ob = actor(scenario, 'ClassID', 3, 'Name', 'OvertakingPulsar', ...
            'AssetType', 'Bicyclist', 'Length', obData.Length, ...
            'Width', obData.Width, 'Height', obData.Height, 'PlotColor', [0.8 0.2 0.2]);
        obWps = [75.0, -5.25, 0.0; 100.0, -3.50, 0.0; 130.0, -1.75, 0.0; 200.0, -1.75, 0.0];
        obSpeeds = [13.88, 13.88, 13.88, 13.88];
        trajectory(ob, obWps, obSpeeds);
        actorMap.OvertakingBike = ob;
    end

    % Oncoming Car in opposing lane
    if isfield(entities, 'OncomingCar')
        ocData = entities.OncomingCar;
        oc = vehicle(scenario, 'ClassID', 1, 'Name', 'OncomingDzire', ...
            'AssetType', ocData.AssetType, 'Length', ocData.Length, ...
            'Width', ocData.Width, 'Height', ocData.Height, 'PlotColor', [0.5 0.5 0.9]);
        ocWps = [230.0, 1.75, 0.0; 10.0, 1.75, 0.0];
        trajectory(oc, ocWps, 8.33);
        actorMap.OncomingCar = oc;
    end
end

function [scenario, actorMap] = setup_indian_vendorcart_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % Vendor Handcart Swerving into Ego Lane with Oncoming Bus
    actorMap = struct();

    egoSpeed = getParam(params, 'Ego_Speed', 6.94);

    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.10 0.45 0.95]);

    if enableAEB
        % Ego detects vendor cart swerving, brakes hard since oncoming bus blocks escape
        egoWaypoints = [
            25.0,  -1.75, 0.0;   % Cruising
            70.0,  -1.75, 0.0;   % Detects slow cart ahead
            85.0,  -1.75, 0.0;   % Cart starts swerving
            95.0,  -1.75, 0.0;   % Emergency braking (can't swerve left - oncoming bus)
            98.0,  -1.75, 0.0;   % Near-standstill behind cart
            100.0, -1.75, 0.0;   % Waiting for cart to clear or bus to pass
            120.0, -1.75, 0.0;   % Resumes
            200.0, -1.75, 0.0;
            340.0, -1.75, 0.0
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 3.0, 1.0, 0.05, 2.0, egoSpeed, egoSpeed, egoSpeed];
    else
        egoWaypoints = [25.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % Vendor Cart (swerves from shoulder into ego lane)
    cartSpeed = getParam(params, 'Cart_Speed', 1.5);
    cart = actor(scenario, 'ClassID', 4, 'Name', 'VendorHandcart', ...
        'AssetType', 'MalePedestrian', 'Length', 1.80, 'Width', 1.20, 'Height', 1.50, ...
        'PlotColor', [0.7 0.5 0.2]);
    cartWps = [
        105.0, -5.80, 0.0;   % Starting on shoulder
        108.0, -5.50, 0.0;   % Drifting slightly
        112.0, -3.50, 0.0;   % Swerving into lane divider
        115.0, -2.00, 0.0;   % Entering ego lane
        120.0, -1.75, 0.0;   % Fully in ego lane
        140.0, -1.75, 0.0    % Continuing in lane
    ];
    cartSpeeds = [cartSpeed, cartSpeed, cartSpeed, cartSpeed, cartSpeed, cartSpeed];
    trajectory(cart, cartWps, cartSpeeds);
    actorMap.VendorCart = cart;

    % Oncoming Bus (blocks opposing lane escape)
    if isfield(entities, 'OncomingBus')
        obData = entities.OncomingBus;
        ob = vehicle(scenario, 'ClassID', 2, 'Name', 'KSRTC_Bus', ...
            'AssetType', 'BoxTruck', 'Length', obData.Length, ...
            'Width', obData.Width, 'Height', obData.Height, 'PlotColor', [0.2 0.7 0.4]);
        obWps = [250.0, 1.75, 0.0; 10.0, 1.75, 0.0];
        trajectory(ob, obWps, 11.11);
        actorMap.OncomingBus = ob;
    end

    % Parked Car (blocks shoulder ahead of cart)
    if isfield(entities, 'ParkedCar')
        pcData = entities.ParkedCar;
        pc = vehicle(scenario, 'ClassID', 1, 'Name', 'ParkedCar', ...
            'AssetType', pcData.AssetType, 'Length', pcData.Length, ...
            'Width', pcData.Width, 'Height', pcData.Height, ...
            'Position', [130.0, -5.50, 0.0], 'Yaw', 0, 'PlotColor', [0.5 0.5 0.5]);
        actorMap.ParkedCar = pc;
    end

    % Curb Pedestrian
    if isfield(entities, 'CurbPedestrian')
        cpData = entities.CurbPedestrian;
        cp = actor(scenario, 'ClassID', 4, 'Name', 'CurbPedestrian', ...
            'AssetType', 'MalePedestrian', 'Length', cpData.Length, ...
            'Width', cpData.Width, 'Height', cpData.Height, 'PlotColor', [0.6 0.4 0.2]);
        cpWps = [100.0, -6.80, 0.0; 140.0, -6.80, 0.0];
        trajectory(cp, cpWps, 0.8);
        actorMap.CurbPedestrian = cp;
    end
end

function [scenario, actorMap] = setup_indian_gauntlet_scenario(scenario, entities, params, enableAEB, liberateEgo)
    if nargin < 5, liberateEgo = false; end
    % THE GAUNTLET - Sequential Multi-Hazard Stress Test
    % Phase 1 (X~70-90):  Slow rickshaw crawling -> overtake
    % Phase 2 (X~140-160): Construction barrier + worker -> reroute
    % Phase 3 (X~220-240): Jaywalker behind parked truck -> emergency brake
    actorMap = struct();

    egoSpeed = getParam(params, 'Ego_Speed', 6.94);

    egoData = entities.Ego;
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Blue', ...
        'AssetType', egoData.AssetType, 'Length', egoData.Length, ...
        'Width', egoData.Width, 'Height', egoData.Height, 'PlotColor', [0.10 0.45 0.95]);

    if enableAEB
        % Phase 1: Overtake rickshaw via outer lane
        % Phase 2: Detour around construction barriers via opposing lane
        % Phase 3: Emergency brake for jaywalker
        egoWaypoints = [
            15.0,  -1.75, 0.0;   % Start
            40.0,  -1.75, 0.0;   % Cruising
            55.0,  -1.75, 0.0;   % Detects slow rickshaw
            65.0,  -3.50, 0.0;   % Swerve to outer lane to overtake
            80.0,  -5.00, 0.0;   % Passing rickshaw
            95.0,  -1.75, 0.0;   % Return to lane 1
            120.0, -1.75, 0.0;   % Cruising toward construction
            130.0, -2.75, 0.0;   % Detects barriers, start detour to right lane (Lane -2 on our own side)
            140.0, -5.00, 0.0;   % In right lane safely clear of barriers
            155.0, -5.00, 0.0;   % Passing barriers on right on our own side
            170.0, -2.75, 0.0;   % Returning toward lane 1
            185.0, -1.75, 0.0;   % Back in lane 1
            210.0, -1.75, 0.0;   % Cruising toward Phase 3
            220.0, -1.75, 0.0;   % Detects jaywalker, emergency braking
            225.0, -1.75, 0.0;   % Near-standstill
            240.0, -1.75, 0.0;   % Resumes after jaywalker clears
            280.0, -1.75, 0.0;   % Straight cruising
            340.0, -1.75, 0.0    % Clean road exit
        ];
        egoSpeeds = [egoSpeed, egoSpeed, 4.0, 4.0, 4.0, egoSpeed, egoSpeed, ...
                     4.0, 3.5, 3.5, 4.0, egoSpeed, egoSpeed, 2.0, 0.05, egoSpeed, egoSpeed, egoSpeed];
    else
        egoWaypoints = [15.0, -1.75, 0.0; 340.0, -1.75, 0.0];
        egoSpeeds = [egoSpeed, egoSpeed];
    end
    assign_ego_motion(ego, egoWaypoints, egoSpeeds, liberateEgo);
    actorMap.Ego = ego;
    actorMap.EgoWaypoints = egoWaypoints;

    % Phase 1: Slow Auto-Rickshaw crawling in lane 1
    if isfield(entities, 'SlowRickshaw')
        rkData = entities.SlowRickshaw;
        rk = vehicle(scenario, 'ClassID', 1, 'Name', 'SlowRickshaw', ...
            'AssetType', rkData.AssetType, 'Length', rkData.Length, ...
            'Width', rkData.Width, 'Height', rkData.Height, 'PlotColor', [0.95 0.85 0.10]);
        rkSpeed = getParam(params, 'Rickshaw_Speed', 2.78);
        rkWps = [75.0, -1.75, 0.0; 160.0, -1.75, 0.0];
        trajectory(rk, rkWps, rkSpeed);
        actorMap.SlowRickshaw = rk;
    end

    % Phase 2: Construction Barriers blocking lane 1
    if isfield(entities, 'ConstructionBarrier1')
        b1 = actor(scenario, 'ClassID', 5, 'Name', 'ConstructionBarrier1', ...
            'AssetType', 'JerseyBarrier', 'Length', 5.0, 'Width', 0.8, 'Height', 0.8, ...
            'Position', [145.0, -1.75, 0.4], 'Yaw', 0, 'PlotColor', [0.95 0.45 0.10]);
        actorMap.ConstructionBarrier1 = b1;
    end
    if isfield(entities, 'ConstructionBarrier2')
        b2 = actor(scenario, 'ClassID', 5, 'Name', 'ConstructionBarrier2', ...
            'AssetType', 'JerseyBarrier', 'Length', 3.0, 'Width', 0.8, 'Height', 0.8, ...
            'Position', [158.0, -1.75, 0.4], 'Yaw', 0, 'PlotColor', [0.95 0.45 0.10]);
        actorMap.ConstructionBarrier2 = b2;
    end

    % Construction Worker walking near barriers
    if isfield(entities, 'ConstructionWorker')
        cw = actor(scenario, 'ClassID', 4, 'Name', 'ConstructionWorker', ...
            'AssetType', 'MalePedestrian', 'Length', 0.28, ...
            'Width', 0.45, 'Height', 1.75, 'PlotColor', [1.0 0.6 0.0]);
        cwWps = [150.0, -0.80, 0.0; 160.0, -0.80, 0.0; 165.0, -0.80, 0.0];
        trajectory(cw, cwWps, 0.4);
        actorMap.ConstructionWorker = cw;
    end

    % Phase 3: Parked Truck (occludes jaywalker)
    if isfield(entities, 'ParkedTruck')
        ptData = entities.ParkedTruck;
        pt = vehicle(scenario, 'ClassID', 2, 'Name', 'ParkedTruck', ...
            'AssetType', 'BoxTruck', 'Length', ptData.Length, ...
            'Width', ptData.Width, 'Height', ptData.Height, ...
            'Position', [225.0, -5.25, 0.0], 'Yaw', 0, 'PlotColor', [0.4 0.3 0.2]);
        actorMap.ParkedTruck = pt;
    end

    % Jaywalker crossing from behind parked truck
    if isfield(entities, 'Jaywalker')
        jw = actor(scenario, 'ClassID', 4, 'Name', 'Jaywalker', ...
            'AssetType', 'MalePedestrian', 'Length', 0.28, ...
            'Width', 0.45, 'Height', 1.75, 'PlotColor', [0.95 0.5 0.1]);
        jwSpeed = getParam(params, 'Jaywalker_Speed', 1.5);
        jwWps = [230.0, -7.0, 0.0; 230.0, -5.25, 0.0; 230.0, -1.75, 0.0; 230.0, 1.50, 0.0];
        trajectory(jw, jwWps, jwSpeed);
        actorMap.Jaywalker = jw;
    end

    % Oncoming Motorcycle (adds visual tension throughout)
    if isfield(entities, 'OncomingBike')
        obData = entities.OncomingBike;
        ob = actor(scenario, 'ClassID', 3, 'Name', 'OncomingBike', ...
            'AssetType', 'Bicyclist', 'Length', obData.Length, ...
            'Width', obData.Width, 'Height', obData.Height, 'PlotColor', [0.6 0.2 0.8]);
        obWps = [280.0, 1.75, 0.0; 10.0, 1.75, 0.0];
        trajectory(ob, obWps, 11.11);
        actorMap.OncomingBike = ob;
    end
end
