use Cwd;

$out_dir   = 'build';
$pdf_mode  = 5;          # xelatex
$xelatex   = 'xelatex -interaction=nonstopmode %O %S';

my $project_dir = getcwd();
$project_dir =~ s{^/([a-zA-Z])/}{\u$1:/};
$ENV{'BIBINPUTS'} = $project_dir . ';' . ($ENV{'BIBINPUTS'} // '');
