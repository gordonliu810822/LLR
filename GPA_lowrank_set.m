function options = GPA_lowrank_set(nGWAS,opts)

%--------------------------------------------------------------------------
% GPAgaussSet creates or alters an options structure for GPAgauss.m.
%--------------------------------------------------------------------------
%   options = GPAgaussSet; (with no input arguments)
%   creates a structure with all fields set to their default values.
%   Each field is an option (also called a parameter).
%
%   GPAgaussSet (with no input or output arguments)
%   displays all options and their default values.
%
%   options = GPAgaussSet(opts); 
%   creates a structure with all fields set to their default values,
%   except valid fields in the structure "opts" replace the defaults.
%
% options.verbose     indicator whether the output is shown in each
%                     iteration, the default is 1.
% options.maxIters    The maximum number of iterations, default is 2000.

% Set default options.
options.initBetaMean = 0.1;
options.initPi = 0.1;

options.verbose = 1;
options.innerMaxIters = 1;
options.maxIters = 2000;
options.eps = 0.5;
options.epsStopLogLik = 1e-5;
%     opts.initBetaMean = 0.1;
%     opts.initPi = 0.1;
%
options.lbPi1 = 0.001;
options.lbBetaAlpha = 0.001;
options.epsilon = 0.5;
options.lam = 10;
options.nlam = 10;
options.maxlam = 100;
options.alpha0 = ones(nGWAS,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End default options.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Quick return if no user opts
  if nargin == 0 || isempty(opts)
    if nargout == 0    % Display options.
      disp('pdco default options:')
      disp( options )
    end
    return
  end

% List of valid field names
  vfields = fieldnames( options );

% Grab valid fields from user's opts
  for i = 1:length(vfields)
    field = vfields{i};
    if isfield( opts, field );
      options.(field) = opts.(field);
    end
  end
