files_str=$(git diff --name-only HEAD HEAD~1)
commit_message=$(git show -s --format=%B HEAD)

echo $files_str

files=$(echo $files_str | tr " " "\n")

echo Commit message : $commit_message 

for file in $files
do
    echo $file
    cp -r $file ../seerslab_bitbucket/seersegmentation/
done

cd ../seerslab_bitbucket/seersegmentation/

git add -A .
git commit -m "$commit_message"
git push origin master

